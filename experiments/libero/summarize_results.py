import os
import json
import argparse
from collections import defaultdict
import pandas as pd
import math
from collections.abc import Mapping, Sequence


_ROUTES = ("wo", "w")
_LATENCY_FIELDS = {
    "gate": ("gate_latency_s",),
    "policy": ("policy_latency_s", "model_latency_s"),
    "total": ("total_latency_s", "query_latency_s"),
}


def _is_record_sequence(value):
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _finite_nonnegative(value, *, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a real number")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{field} must be finite and non-negative")
    return value


def _percentile(values, quantile):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] + fraction * (ordered[upper] - ordered[lower]))


def _extract_query_payload(query, *, field):
    if not isinstance(query, Mapping):
        raise TypeError(f"{field} must be a mapping")
    for nested_key in ("decision", "routing_decision"):
        nested = query.get(nested_key)
        if isinstance(nested, Mapping):
            merged = dict(nested)
            merged.update({key: value for key, value in query.items() if key != nested_key})
            return merged
    return query


def _query_records_and_episode_totals(result):
    """Return exact query records, preferring episode-scoped telemetry.

    The evaluator's canonical shape is ``routing.episodes[*].queries``. The
    aliases keep the summarizer compatible with early development artifacts
    without ever counting two copies of the same query list.
    """

    routing = result.get("routing")
    if isinstance(routing, Mapping):
        episodes = routing.get("episodes")
        if _is_record_sequence(episodes):
            records = []
            episode_totals = []
            found_query_list = False
            for episode_index, episode in enumerate(episodes):
                if not isinstance(episode, Mapping):
                    raise TypeError(f"routing.episodes[{episode_index}] must be a mapping")
                queries = episode.get("queries", episode.get("query_metrics"))
                if queries is None:
                    continue
                if not _is_record_sequence(queries):
                    raise TypeError(
                        f"routing.episodes[{episode_index}].queries must be a sequence"
                    )
                found_query_list = True
                episode_records = [
                    _extract_query_payload(
                        query,
                        field=f"routing.episodes[{episode_index}].queries[{query_index}]",
                    )
                    for query_index, query in enumerate(queries)
                ]
                records.extend(episode_records)
                episode_totals.append(
                    sum(int(query.get("actual_video_steps", query.get("selected_video_nfe", 0)))
                        for query in episode_records)
                )
            if found_query_list:
                return records, episode_totals

        for key in ("query_metrics", "queries"):
            queries = routing.get(key)
            if _is_record_sequence(queries):
                records = [
                    _extract_query_payload(query, field=f"routing.{key}[{index}]")
                    for index, query in enumerate(queries)
                ]
                grouped = defaultdict(int)
                has_episode_ids = False
                for query in records:
                    if query.get("episode_index") is not None:
                        has_episode_ids = True
                        grouped[int(query["episode_index"])] += int(
                            query.get("actual_video_steps", query.get("selected_video_nfe", 0))
                        )
                return records, list(grouped.values()) if has_episode_ids else []

    for key in ("routing_query_metrics", "query_metrics"):
        queries = result.get(key)
        if _is_record_sequence(queries):
            return [
                _extract_query_payload(query, field=f"{key}[{index}]")
                for index, query in enumerate(queries)
            ], []
    return None, []


def _routing_summary_payload(result):
    routing = result.get("routing")
    if isinstance(routing, Mapping):
        summary = routing.get("summary")
        if isinstance(summary, Mapping):
            return summary
        if isinstance(routing.get("counts"), Mapping):
            return routing
    summary = result.get("routing_summary")
    return summary if isinstance(summary, Mapping) else None


def _reported_episode_video_nfe(result):
    routing = result.get("routing")
    if not isinstance(routing, Mapping):
        return []
    episodes = routing.get("episodes")
    if not _is_record_sequence(episodes):
        return []
    totals = []
    for episode_index, episode in enumerate(episodes):
        if not isinstance(episode, Mapping):
            raise TypeError(f"routing.episodes[{episode_index}] must be a mapping")
        value = episode.get("total_actual_video_steps")
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                f"routing.episodes[{episode_index}].total_actual_video_steps must be a non-negative integer"
            )
        totals.append(value)
    return totals


def aggregate_routing_results(results):
    """Merge routing telemetry by total policy-query count.

    Raw per-query records are authoritative and give exact percentiles. Older
    task summaries are accepted as a fallback; their means are query-weighted,
    while non-mergeable task percentiles are deliberately reported as null.
    """

    counts = {"total": 0, "wo": 0, "w": 0}
    total_video_nfe = 0
    episode_video_nfe = []
    raw_latencies = {name: [] for name in _LATENCY_FIELDS}
    latency_sums = {name: 0.0 for name in _LATENCY_FIELDS}
    latency_counts = {name: 0 for name in _LATENCY_FIELDS}
    raw_tasks = 0
    fallback_tasks = 0

    for result_index, result in enumerate(results):
        if not isinstance(result, Mapping):
            raise TypeError(f"results[{result_index}] must be a mapping")
        queries, task_episode_totals = _query_records_and_episode_totals(result)
        reported_episode_totals = _reported_episode_video_nfe(result)
        if reported_episode_totals:
            if task_episode_totals and task_episode_totals != reported_episode_totals:
                raise ValueError(
                    f"results[{result_index}] per-episode video NFE disagrees with raw queries"
                )
            task_episode_totals = reported_episode_totals
        task_summary = _routing_summary_payload(result)
        if queries == [] and isinstance(task_summary, Mapping):
            summary_counts = task_summary.get("counts", {})
            if isinstance(summary_counts, Mapping) and int(summary_counts.get("total", 0)) > 0:
                # ``save_query_metrics=false`` may retain an empty list next
                # to a non-empty exact task summary. Do not erase its totals.
                queries = None
        if queries is not None:
            raw_tasks += 1
            episode_video_nfe.extend(task_episode_totals)
            for query_index, query in enumerate(queries):
                route = query.get("selected_mode", query.get("route"))
                if route not in _ROUTES:
                    raise ValueError(
                        f"results[{result_index}] query {query_index} has invalid selected_mode={route!r}"
                    )
                steps = query.get("actual_video_steps", query.get("selected_video_nfe"))
                if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
                    raise ValueError(
                        f"results[{result_index}] query {query_index} has invalid actual_video_steps"
                    )
                if route == "wo" and steps != 0:
                    raise ValueError("wo routing query cannot report non-zero video NFE")
                counts["total"] += 1
                counts[route] += 1
                total_video_nfe += steps

                # Warmup queries still contribute to the realized route rate
                # and compute budget, but are excluded from latency statistics.
                if query.get("timing_included", True) is False:
                    continue
                observed = {}
                for latency_name, aliases in _LATENCY_FIELDS.items():
                    value = next((query.get(key) for key in aliases if query.get(key) is not None), None)
                    if value is not None:
                        observed[latency_name] = _finite_nonnegative(
                            value,
                            field=f"results[{result_index}] query {query_index} {latency_name} latency",
                        )
                if "total" not in observed and "gate" in observed and "policy" in observed:
                    observed["total"] = observed["gate"] + observed["policy"]
                for latency_name, value in observed.items():
                    raw_latencies[latency_name].append(value)
                    latency_sums[latency_name] += value
                    latency_counts[latency_name] += 1
            continue

        summary = task_summary
        if summary is None:
            continue
        fallback_tasks += 1
        episode_video_nfe.extend(task_episode_totals)
        task_counts = summary.get("counts")
        if not isinstance(task_counts, Mapping):
            raise ValueError(f"results[{result_index}] routing summary is missing counts")
        task_total = int(task_counts.get("total", 0))
        task_wo = int(task_counts.get("wo", 0))
        task_w = int(task_counts.get("w", 0))
        if min(task_total, task_wo, task_w) < 0 or task_wo + task_w != task_total:
            raise ValueError(f"results[{result_index}] routing counts are inconsistent")
        effective = summary.get("effective_video_steps", summary.get("actual_video_nfe", {}))
        if not isinstance(effective, Mapping):
            raise ValueError(f"results[{result_index}] routing summary is missing video NFE")
        task_video_nfe = int(effective.get("total", effective.get("actual_total_video_nfe", 0)))
        if task_video_nfe < 0:
            raise ValueError(f"results[{result_index}] routing video NFE is negative")
        counts["total"] += task_total
        counts["wo"] += task_wo
        counts["w"] += task_w
        total_video_nfe += task_video_nfe

        latency = summary.get("latency_s", {})
        if isinstance(latency, Mapping):
            for latency_name in _LATENCY_FIELDS:
                item = latency.get(latency_name)
                if isinstance(item, Mapping) and item.get("mean") is not None:
                    mean = _finite_nonnegative(
                        item["mean"],
                        field=f"results[{result_index}] routing {latency_name} mean",
                    )
                    sample_count = int(item.get("count", task_total))
                    if sample_count < 0:
                        raise ValueError("routing latency count cannot be negative")
                    latency_sums[latency_name] += mean * sample_count
                    latency_counts[latency_name] += sample_count

    if raw_tasks == 0 and fallback_tasks == 0:
        return None

    latency_output = {}
    for latency_name in _LATENCY_FIELDS:
        values = raw_latencies[latency_name]
        count = latency_counts[latency_name]
        latency_output[latency_name] = {
            "count": count,
            "mean": float(latency_sums[latency_name] / count) if count else None,
            "p50": _percentile(values, 0.50) if fallback_tasks == 0 else None,
            "p95": _percentile(values, 0.95) if fallback_tasks == 0 else None,
        }

    total_queries = counts["total"]
    mean_video_nfe = float(total_video_nfe / total_queries) if total_queries else 0.0
    if raw_tasks and fallback_tasks:
        source = "mixed"
    elif raw_tasks:
        source = "query_records"
    else:
        source = "task_summaries_fallback"
    per_episode_video_nfe = {
        "count": len(episode_video_nfe),
        "mean": (
            float(math.fsum(episode_video_nfe) / len(episode_video_nfe))
            if episode_video_nfe
            else None
        ),
        "p50": _percentile(episode_video_nfe, 0.50),
        "p95": _percentile(episode_video_nfe, 0.95),
    }
    return {
        "source": source,
        "counts": counts,
        "with_rate": float(counts["w"] / total_queries) if total_queries else 0.0,
        "actual_total_video_nfe": int(total_video_nfe),
        "avg_video_nfe_per_query": mean_video_nfe,
        "effective_video_steps": {
            "total": int(total_video_nfe),
            "mean": mean_video_nfe,
        },
        "per_episode_video_nfe": per_episode_video_nfe,
        "total_video_nfe_per_episode": per_episode_video_nfe,
        "latency_s": latency_output,
        "latency_percentiles_exact": fallback_tasks == 0,
    }

def format_time(seconds):
    """Format seconds as a human-readable duration string.

    - Below 1 minute: SS
    - Below 1 hour: MMSS
    - 1 hour or longer: HHMMSS
    """
    seconds = round(seconds)  # Round to integer seconds
    
    if seconds < 60:
        return f"{seconds:02d}s"
    elif seconds < 3600:
        minutes = seconds // 60
        remaining_seconds = seconds % 60
        return f"{minutes:02d}m{remaining_seconds:02d}s"
    else:
        hours = seconds // 3600
        remaining = seconds % 3600
        minutes = remaining // 60
        remaining_seconds = remaining % 60
        return f"{hours:02d}h{minutes:02d}m{remaining_seconds:02d}s"

def summarize_results(output_dir):
    """Summarize all evaluation results.

    Args:
        output_dir: Root directory containing result files.
    """
    # Store statistics for each suite
    suite_stats = defaultdict(lambda: {
        'total_tasks': 0,
        'total_trials': 0,
        'total_successes': 0,
        'total_time': 0,
        'max_time': 0,
        'psnr_sum': 0.0,
        'psnr_count': 0
    })
    
    # Store detailed per-task results
    task_results = {}
    has_psnr_metric = False
    routing_results_by_suite = defaultdict(list)
    all_routing_results = []
    
    # Iterate over all suite directories
    for suite in ["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"]:
        suite_dir = os.path.join(output_dir, suite)
        if not os.path.exists(suite_dir):
            continue
            
        # Read all result files
        for filename in os.listdir(suite_dir):
            if not filename.startswith('gpu') or not filename.endswith('_results.json'):
                continue
                
            with open(os.path.join(suite_dir, filename), 'r') as f:
                result = json.load(f)

            task_routing = aggregate_routing_results([result])
            if task_routing is not None:
                routing_results_by_suite[suite].append(result)
                all_routing_results.append(result)
            
            # Extract task ID from the filename
            parts = filename.split('_')
            task_id = int(parts[1].replace('task', ''))
            
            # Create the task identifier (suite_taskid)
            task_key = f"{suite}_{task_id}"
                
            stats = suite_stats[suite]
            stats['total_tasks'] += 1
            stats['total_trials'] += result['total_episodes']
            stats['total_successes'] += result['successes']
            stats['total_time'] += result['duration']
            stats['max_time'] = max(stats['max_time'], result['duration'])
            if 'future_video_psnr_mean' in result:
                has_psnr_metric = True
                if result['future_video_psnr_mean'] is not None:
                    stats['psnr_sum'] += float(result['future_video_psnr_mean'])
                    stats['psnr_count'] += 1
            
            # Store detailed task results
            task_result = {
                'success_rate': result['successes'] / result['total_episodes'] * 100,
                'duration': result['duration'],
                'total_episodes': result['total_episodes'],
                'successes': result['successes'],
                'task_description': result['task_description'] if 'task_description' in result else ''
            }
            if 'future_video_psnr_mean' in result:
                task_result['future_video_psnr_mean'] = (
                    float(result['future_video_psnr_mean'])
                    if result['future_video_psnr_mean'] is not None
                    else None
                )
            if task_routing is not None:
                task_result['routing'] = task_routing
            task_results[task_key] = task_result

    suite_routing = {
        suite: aggregate_routing_results(results)
        for suite, results in routing_results_by_suite.items()
    }
    overall_routing = aggregate_routing_results(all_routing_results)
    has_routing_metric = overall_routing is not None
    
    # Print summary results
    print("\n=== Evaluation Results Summary ===")
    print("\nStatistics for each task suite:")
    
    total_success_rate = 0
    total_time = 0
    total_suites = 0
    overall_psnr_sum = 0.0
    overall_psnr_count = 0
    
    # Prepare DataFrame rows
    df_data = {
        'Task Suite': [],
        'Success Rate (%)': [],
        'Average Time (s)': [],
        'Max Time (s)': []
    }
    if has_psnr_metric:
        df_data['Average Future PSNR (dB)'] = []
    if has_routing_metric:
        df_data['With-video Rate (%)'] = []
        df_data['Average Video NFE / Query'] = []
        df_data['Average Query Latency (s)'] = []
    
    for suite, stats in suite_stats.items():
        if stats['total_trials'] > 0:
            success_rate = stats['total_successes'] / stats['total_trials'] * 100
            avg_time = stats['total_time'] / stats['total_tasks']
            max_time = stats['max_time']
            suite_avg_psnr = None
            if has_psnr_metric:
                suite_avg_psnr = (
                    stats['psnr_sum'] / stats['psnr_count']
                    if stats['psnr_count'] > 0
                    else None
                )
            
            print(f"\n{suite}:")
            print(f"- Tasks completed: {stats['total_tasks']}")
            print(f"- Total attempts: {stats['total_trials']}")
            print(f"- Successful attempts: {stats['total_successes']}")
            print(f"- Success rate: {success_rate:.2f}%")
            print(f"- Total time: {format_time(stats['total_time'])}")
            print(f"- Average time per task: {format_time(avg_time)}")
            print(f"- Longest task time: {format_time(max_time)}")
            if has_psnr_metric:
                if suite_avg_psnr is not None:
                    print(f"- Average future-video PSNR: {suite_avg_psnr:.4f} dB")
                else:
                    print("- Average future-video PSNR: N/A")
            routing = suite_routing.get(suite)
            if has_routing_metric:
                if routing is not None:
                    print(f"- Policy queries: {routing['counts']['total']}")
                    print(f"- Routes (wo/w): {routing['counts']['wo']}/{routing['counts']['w']}")
                    print(f"- With-video rate: {100.0 * routing['with_rate']:.2f}%")
                    print(f"- Actual total video NFE: {routing['actual_total_video_nfe']}")
                    print(f"- Average video NFE/query: {routing['avg_video_nfe_per_query']:.6f}")
                    query_latency = routing['latency_s']['total']['mean']
                    print(
                        "- Average query latency: "
                        + (f"{query_latency:.6f}s" if query_latency is not None else "N/A")
                    )
                else:
                    print("- Routing telemetry: N/A")
            
            # Append to DataFrame rows
            df_data['Task Suite'].append(suite)
            df_data['Success Rate (%)'].append(f"{success_rate:.2f}")
            df_data['Average Time (s)'].append(f"{avg_time:.2f}")
            df_data['Max Time (s)'].append(f"{max_time:.2f}")
            if has_psnr_metric:
                df_data['Average Future PSNR (dB)'].append(
                    f"{suite_avg_psnr:.4f}" if suite_avg_psnr is not None else "N/A"
                )
            if has_routing_metric:
                df_data['With-video Rate (%)'].append(
                    f"{100.0 * routing['with_rate']:.2f}" if routing is not None else "N/A"
                )
                df_data['Average Video NFE / Query'].append(
                    f"{routing['avg_video_nfe_per_query']:.6f}" if routing is not None else "N/A"
                )
                query_latency = (
                    routing['latency_s']['total']['mean'] if routing is not None else None
                )
                df_data['Average Query Latency (s)'].append(
                    f"{query_latency:.6f}" if query_latency is not None else "N/A"
                )
            
            total_success_rate += success_rate
            total_time += stats['total_time']
            total_suites += 1
            if has_psnr_metric:
                overall_psnr_sum += stats['psnr_sum']
                overall_psnr_count += stats['psnr_count']
    
    if total_suites > 0:
        print("\nOverall statistics:")
        avg_success_rate = total_success_rate/total_suites
        avg_task_time = total_time/sum(s['total_tasks'] for s in suite_stats.values())
        max_task_time = max(s['max_time'] for s in suite_stats.values())
        overall_avg_psnr = None
        if has_psnr_metric:
            overall_avg_psnr = overall_psnr_sum / overall_psnr_count if overall_psnr_count > 0 else None
        
        print(f"- Average success rate: {avg_success_rate:.2f}%")
        print(f"- Total time: {format_time(total_time)}")
        print(f"- Average time per task: {format_time(avg_task_time)}")
        print(f"- Longest task time: {format_time(max_task_time)}")
        if has_psnr_metric:
            if overall_avg_psnr is not None:
                print(f"- Average future-video PSNR: {overall_avg_psnr:.4f} dB")
            else:
                print("- Average future-video PSNR: N/A")
        if overall_routing is not None:
            print(f"- Policy queries: {overall_routing['counts']['total']}")
            print(
                "- Routes (wo/w): "
                f"{overall_routing['counts']['wo']}/{overall_routing['counts']['w']}"
            )
            print(f"- With-video rate: {100.0 * overall_routing['with_rate']:.2f}%")
            print(f"- Actual total video NFE: {overall_routing['actual_total_video_nfe']}")
            print(
                "- Average video NFE/query: "
                f"{overall_routing['avg_video_nfe_per_query']:.6f}"
            )
        
        # Add an overall summary row
        df_data['Task Suite'].append('Overall')
        df_data['Success Rate (%)'].append(f"{avg_success_rate:.2f}")
        df_data['Average Time (s)'].append(f"{avg_task_time:.2f}")
        df_data['Max Time (s)'].append(f"{max_task_time:.2f}")
        if has_psnr_metric:
            df_data['Average Future PSNR (dB)'].append(
                f"{overall_avg_psnr:.4f}" if overall_avg_psnr is not None else "N/A"
            )
        if has_routing_metric:
            df_data['With-video Rate (%)'].append(
                f"{100.0 * overall_routing['with_rate']:.2f}"
            )
            df_data['Average Video NFE / Query'].append(
                f"{overall_routing['avg_video_nfe_per_query']:.6f}"
            )
            overall_query_latency = overall_routing['latency_s']['total']['mean']
            df_data['Average Query Latency (s)'].append(
                f"{overall_query_latency:.6f}" if overall_query_latency is not None else "N/A"
            )
    
    # Create and save the DataFrame
    df = pd.DataFrame(df_data)
    
    # Use the last checkpoint path component as the title
    ckpt_path = os.environ.get('CKPT', '')
    title = os.path.basename(ckpt_path) if ckpt_path else 'Results'
    
    # Transpose the DataFrame and use Task Suite as column names
    df = df.set_index('Task Suite').T
    
    # Add a title line to the CSV file
    with open(os.path.join(output_dir, 'summary.csv'), 'w') as f:
        f.write(f"{title}\n")  # Write the title
        df.to_csv(f)
    
    # Create the per-task success-rate CSV
    task_success_data = {
        'Task': [],
        'Description': [],
        'Success Rate (%)': []
    }
    if has_psnr_metric:
        task_success_data['Future Video PSNR (dB)'] = []
    if has_routing_metric:
        task_success_data['With-video Rate (%)'] = []
        task_success_data['Average Video NFE / Query'] = []
    
    # Group tasks by suite
    suite_tasks = defaultdict(list)
    for task in task_results:
        suite = task.split('_')[0] + '_' + task.split('_')[1]
        suite_tasks[suite].append(task)
    
    # Sort tasks within each suite
    for suite in suite_tasks:
        suite_tasks[suite].sort(key=lambda x: int(x.split('_')[-1]))
    
    # Fill per-task success-rate rows
    for suite in sorted(suite_tasks.keys()):
        for task in suite_tasks[suite]:
            result = task_results[task]
            task_success_data['Task'].append(task)
            task_success_data['Description'].append(
                result['task_description'] if 'task_description' in result else ''
            )
            task_success_data['Success Rate (%)'].append(f"{result['success_rate']:.2f}")
            if has_psnr_metric:
                psnr = result['future_video_psnr_mean'] if 'future_video_psnr_mean' in result else None
                task_success_data['Future Video PSNR (dB)'].append(
                    f"{psnr:.4f}" if psnr is not None else "N/A"
                )
            if has_routing_metric:
                routing = result.get('routing')
                task_success_data['With-video Rate (%)'].append(
                    f"{100.0 * routing['with_rate']:.2f}" if routing is not None else "N/A"
                )
                task_success_data['Average Video NFE / Query'].append(
                    f"{routing['avg_video_nfe_per_query']:.6f}" if routing is not None else "N/A"
                )

    suite_stats_output = {}
    for suite, stats in suite_stats.items():
        suite_stats_output[suite] = {
            'total_tasks': stats['total_tasks'],
            'total_trials': stats['total_trials'],
            'total_successes': stats['total_successes'],
            'total_time': stats['total_time'],
            'max_time': stats['max_time'],
        }
        if has_psnr_metric:
            suite_stats_output[suite]['average_future_video_psnr'] = (
                stats['psnr_sum'] / stats['psnr_count'] if stats['psnr_count'] > 0 else None
            )
        if suite_routing.get(suite) is not None:
            suite_stats_output[suite]['routing'] = suite_routing[suite]
    
    # Create and save the task success-rate DataFrame
    task_success_df = pd.DataFrame(task_success_data)
    task_success_df.to_csv(os.path.join(output_dir, 'task_success_rates.csv'), index=False)
    
    # Save the detailed JSON summary
    summary_file = os.path.join(output_dir, 'summary.json')
    overall_stats = {
        'average_success_rate': total_success_rate/total_suites if total_suites > 0 else 0,
        'total_time': total_time,
        'average_task_time': total_time/sum(s['total_tasks'] for s in suite_stats.values()) if suite_stats else 0,
    }
    if has_psnr_metric:
        overall_stats['average_future_video_psnr'] = (
            overall_psnr_sum / overall_psnr_count if overall_psnr_count > 0 else None
        )
    if overall_routing is not None:
        overall_stats['routing'] = overall_routing

    with open(summary_file, 'w') as f:
        json.dump({
            'run_id': os.path.basename(output_dir),
            'ckpt': os.environ.get('CKPT', ''), # Checkpoint path
            'config': os.environ.get('CONFIG', ''), # Config path
            'suite_stats': suite_stats_output,
            'task_results': task_results,
            'overall': overall_stats
        }, f, indent=4)
    
    print(f"\n=== Run Information ===")
    print(f"Run ID: {os.path.basename(output_dir)}")
    print(f"Results directory: {output_dir}")
    print(f"Summary file: {summary_file}")
    print(f"Summary CSV: {os.path.join(output_dir, 'summary.csv')}")
    print(f"Task success rates CSV: {os.path.join(output_dir, 'task_success_rates.csv')}")
    
    # Print the task success-rate table
    print("\n=== Task Success Rates ===")
    print(task_success_df.to_string(index=False))

    # Print the transposed summary table
    print("\n=== Results Table ===")
    print(df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Root directory containing evaluation results')
    args = parser.parse_args()
    
    summarize_results(args.output_dir)

if __name__ == '__main__':
    main() 
