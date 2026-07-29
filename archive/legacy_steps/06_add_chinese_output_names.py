from pathlib import Path
import shutil

import pandas as pd


try:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
except NameError:
    # Helps if the whole file is run in Spyder/IPython instead of as a script.
    PROJECT_ROOT = Path.cwd().resolve()
    if PROJECT_ROOT.name == "code":
        PROJECT_ROOT = PROJECT_ROOT.parent

DATA_PROCESSED = PROJECT_ROOT / "data_processed"
NOTES_DIR = PROJECT_ROOT / "notes"
OUTPUT_DIR = DATA_PROCESSED / "中文命名结果文件"
README_PATH = OUTPUT_DIR / "文件说明_README.csv"


def get_file_mapping():
    """Return the list of files to copy with Chinese explanatory names."""
    return [
        {
            "stage": "01",
            "script": "code/01_inspect_2025_data.py",
            "original_file": "data_processed/matches_2025_checked.csv",
            "chinese_named_file": "01_数据检查_2025比赛合并表_matches_2025_checked.csv",
            "description": (
                "这是 2025 年检查并合并后的逐场比赛数据表，合并了 games、events、"
                "hidx 和 names，用于后续 Elo baseline。"
            ),
        },
        {
            "stage": "02",
            "script": "code/02_simple_elo_2025.py",
            "original_file": "data_processed/elo_predictions_2025.csv",
            "chinese_named_file": "02_Elo逐场预测结果_默认参数_2025_elo_predictions.csv",
            "description": "这是 simple Elo baseline 对每一场比赛的赛前预测和赛后更新记录。",
        },
        {
            "stage": "02",
            "script": "code/02_simple_elo_2025.py",
            "original_file": "data_processed/elo_final_ratings_2025.csv",
            "chinese_named_file": "02_Elo最终评分表_默认参数_2025_final_ratings.csv",
            "description": "这是 2025 年全部比赛跑完后，每个选手的最终 Elo rating。",
        },
        {
            "stage": "02",
            "script": "code/02_simple_elo_2025.py",
            "original_file": "data_processed/elo_scores_2025.csv",
            "chinese_named_file": "02_Elo总体指标_默认参数_2025_scores.csv",
            "description": (
                "这是默认参数 simple Elo 的总体评价指标，包括 log loss、"
                "Brier score 和 accuracy。"
            ),
        },
        {
            "stage": "03",
            "script": "code/03_evaluate_elo_2025.py",
            "original_file": "data_processed/elo_evaluation_2025.csv",
            "chinese_named_file": "03_Elo评估指标复算_2025_evaluation.csv",
            "description": (
                "这是第三阶段重新计算的总体 evaluation metrics，"
                "用来确认第二阶段指标是否一致。"
            ),
        },
        {
            "stage": "03",
            "script": "code/03_evaluate_elo_2025.py",
            "original_file": "data_processed/elo_calibration_2025.csv",
            "chinese_named_file": "03_Elo校准表_按预测概率分箱_2025_calibration.csv",
            "description": (
                "这是按照 pred_a_win 的概率区间生成的 calibration table，"
                "用来检查预测概率和实际胜率是否接近。"
            ),
        },
        {
            "stage": "03",
            "script": "code/03_evaluate_elo_2025.py",
            "original_file": "data_processed/elo_calibration_by_confidence_2025.csv",
            "chinese_named_file": "03_Elo置信度分组表现_2025_confidence_calibration.csv",
            "description": (
                "这是按照模型预测信心 confidence 分组后的 accuracy 表，"
                "用来观察高信心预测是否更准确。"
            ),
        },
        {
            "stage": "04",
            "script": "code/04_elo_parameter_test_2025.py",
            "original_file": "data_processed/elo_parameter_results_2025.csv",
            "chinese_named_file": "04_Elo参数敏感性测试_全部结果_2025_parameter_results.csv",
            "description": (
                "这是不同 K 和 scale 组合下的 Elo 结果汇总，用来比较参数对 "
                "log loss、Brier score 和 accuracy 的影响。"
            ),
        },
        {
            "stage": "05",
            "script": "code/05_summarise_elo_results_2025.py",
            "original_file": "data_processed/elo_summary_default_vs_best_2025.csv",
            "chinese_named_file": "05_Elo汇报汇总_默认参数vs最优参数_2025.csv",
            "description": "这是默认 Elo 和 grid search 最优 Elo 的对比表，适合 meeting 汇报。",
        },
        {
            "stage": "05",
            "script": "code/05_summarise_elo_results_2025.py",
            "original_file": "data_processed/elo_parameter_top10_2025.csv",
            "chinese_named_file": "05_Elo参数Top10_按logloss最好_2025.csv",
            "description": "这是 log loss 最好的前 10 组参数。",
        },
        {
            "stage": "05",
            "script": "code/05_summarise_elo_results_2025.py",
            "original_file": "data_processed/elo_parameter_bottom10_2025.csv",
            "chinese_named_file": "05_Elo参数Bottom10_按logloss最差_2025.csv",
            "description": "这是 log loss 最差的后 10 组参数。",
        },
        {
            "stage": "05",
            "script": "code/05_summarise_elo_results_2025.py",
            "original_file": "data_processed/elo_calibration_nonempty_2025.csv",
            "chinese_named_file": "05_Elo校准表_只保留非空概率区间_2025.csv",
            "description": (
                "这是去掉空 probability bins 之后的 calibration 表，"
                "更适合汇报时查看。"
            ),
        },
        {
            "stage": "05",
            "script": "code/05_summarise_elo_results_2025.py",
            "original_file": "data_processed/elo_confidence_summary_2025.csv",
            "chinese_named_file": "05_Elo置信度汇总表_2025.csv",
            "description": "这是整理后的 confidence-based evaluation summary。",
        },
        {
            "stage": "05",
            "script": "code/05_summarise_elo_results_2025.py",
            "original_file": "notes/meeting2_elo_summary_2025.md",
            "chinese_named_file": "05_meeting2代码汇报总结_2025_Elo_summary.md",
            "description": "这是第二次 meeting 可以使用的代码结果汇报 markdown。",
        },
    ]


def resolve_original_path(original_file):
    """Turn a project-relative source path into an absolute path."""
    return PROJECT_ROOT / original_file


def copy_files_with_chinese_names(file_mapping):
    """Copy existing output files into OUTPUT_DIR using Chinese names."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    copied = []
    missing = []

    for item in file_mapping:
        source_path = resolve_original_path(item["original_file"])
        target_path = OUTPUT_DIR / item["chinese_named_file"]

        if not source_path.exists():
            print(f"WARNING: source file missing, skipped: {source_path}")
            missing.append(item)
            continue

        shutil.copy2(source_path, target_path)
        print(f"Copied: {source_path.name} -> {target_path.name}")
        copied.append(item)

    return copied, missing


def create_readme(file_mapping):
    """Create a README CSV explaining each Chinese-named copy."""
    readme = pd.DataFrame(
        file_mapping,
        columns=[
            "stage",
            "script",
            "original_file",
            "chinese_named_file",
            "description",
        ],
    )
    readme.to_csv(README_PATH, index=False, encoding="utf-8-sig")
    return readme


def print_summary(copied, missing):
    print("\n=== Chinese Output Names Summary ===")
    print(f"成功复制了多少个文件: {len(copied)}")
    print(f"缺失了多少个文件: {len(missing)}")
    print(f"中文命名结果文件夹路径: {OUTPUT_DIR}")
    print(f"README 文件路径: {README_PATH}")


def main():
    file_mapping = get_file_mapping()
    copied, missing = copy_files_with_chinese_names(file_mapping)
    create_readme(file_mapping)
    print_summary(copied, missing)


if __name__ == "__main__":
    main()
