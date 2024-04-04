def combine_ee_eval_results(results):
    max_task = len(results)
    combined_stats_tag = {
        "per_ic_acc": sum(results[i]["per_ic_acc"]["tag"] for i in range(max_task))
        / max_task,
        "per_th_acc": sum(results[i]["per_th_acc"]["tag"] for i in range(max_task))
        / max_task,
        "per_th_exit_cnt": sum(
            results[i]["per_th_exit_cnt"]["tag"] for i in range(max_task)
        )
        / max_task,
    }
    combined_stats_taw = {
        "per_ic_acc": sum(results[i]["per_ic_acc"]["taw"] for i in range(max_task))
        / max_task,
        "per_th_acc": sum(results[i]["per_th_acc"]["taw"] for i in range(max_task))
        / max_task,
        "per_th_exit_cnt": sum(
            results[i]["per_th_exit_cnt"]["taw"] for i in range(max_task)
        )
        / max_task,
    }
    combined_results = {
        "exit_costs": sum(results[i]["exit_costs"] for i in range(max_task)) / max_task,
        "baseline_cost": sum(results[i]["baseline_cost"] for i in range(max_task))
        / max_task,
        "per_ic_acc": {
            "taw": combined_stats_taw["per_ic_acc"],
            "tag": combined_stats_tag["per_ic_acc"],
        },
        "per_th_acc": {
            "taw": combined_stats_taw["per_th_acc"],
            "tag": combined_stats_tag["per_th_acc"],
        },
        "per_th_exit_cnt": {
            "taw": combined_stats_taw["per_th_exit_cnt"],
            "tag": combined_stats_tag["per_th_exit_cnt"],
        },
    }
    return combined_results
