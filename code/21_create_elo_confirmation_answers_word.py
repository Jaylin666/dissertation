"""Create a Word summary answering the Elo confirmation questions."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import html
import re
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_QUESTIONS_DOCX = PROJECT_ROOT.parent / "问elo.docx"
ELO_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "elo_optimization"
OUTPUT_DOCX = ELO_OUTPUT_DIR / "elo_confirmation_answers_summary.docx"


def read_docx_paragraphs(path: Path) -> list[str]:
    with ZipFile(path) as zf:
        xml = zf.read("word/document.xml").decode("utf-8")
    paragraphs: list[str] = []
    for part in xml.split("</w:p>"):
        texts = re.findall(r"<w:t[^>]*>(.*?)</w:t>", part)
        if texts:
            paragraphs.append("".join(html.unescape(text) for text in texts).strip())
    return [paragraph for paragraph in paragraphs if paragraph]


def fmt(value: object, digits: int = 3) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (int,)):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def fmt6(value: object) -> str:
    return fmt(value, 6)


def qxml(text: object) -> str:
    return html.escape("" if text is None else str(text), quote=False)


def paragraph(text: str = "", style: str | None = None, bold: bool = False) -> str:
    style_xml = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
    bold_xml = "<w:b/>" if bold else ""
    return (
        "<w:p>"
        f"{style_xml}"
        "<w:r>"
        f"<w:rPr>{bold_xml}</w:rPr>"
        f"<w:t xml:space=\"preserve\">{qxml(text)}</w:t>"
        "</w:r>"
        "</w:p>"
    )


def bullet(text: str) -> str:
    return paragraph("• " + text)


def table(headers: list[str], rows: list[list[object]]) -> str:
    cells = "".join(
        "<w:tc><w:tcPr><w:tcW w:w=\"2400\" w:type=\"dxa\"/></w:tcPr>"
        f"<w:p><w:r><w:rPr><w:b/></w:rPr><w:t>{qxml(header)}</w:t></w:r></w:p></w:tc>"
        for header in headers
    )
    xml = [
        "<w:tbl>",
        (
            "<w:tblPr>"
            "<w:tblBorders>"
            "<w:top w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"999999\"/>"
            "<w:left w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"999999\"/>"
            "<w:bottom w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"999999\"/>"
            "<w:right w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"999999\"/>"
            "<w:insideH w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"999999\"/>"
            "<w:insideV w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"999999\"/>"
            "</w:tblBorders>"
            "</w:tblPr>"
        ),
        f"<w:tr>{cells}</w:tr>",
    ]
    for row in rows:
        row_xml = "".join(
            "<w:tc><w:tcPr><w:tcW w:w=\"2400\" w:type=\"dxa\"/></w:tcPr>"
            f"<w:p><w:r><w:t xml:space=\"preserve\">{qxml(cell)}</w:t></w:r></w:p></w:tc>"
            for cell in row
        )
        xml.append(f"<w:tr>{row_xml}</w:tr>")
    xml.append("</w:tbl>")
    return "".join(xml)


def make_docx(paragraph_xml: list[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body>"
        + "".join(paragraph_xml)
        + '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1440" w:right="1440" '
        'w:bottom="1440" w:left="1440" w:header="720" w:footer="720" w:gutter="0"/></w:sectPr>'
        "</w:body></w:document>"
    )
    styles_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        '<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/>'
        '<w:rPr><w:sz w:val="22"/><w:szCs w:val="22"/></w:rPr></w:style>'
        '<w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/>'
        '<w:rPr><w:b/><w:sz w:val="32"/><w:szCs w:val="32"/></w:rPr></w:style>'
        '<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/>'
        '<w:basedOn w:val="Normal"/><w:next w:val="Normal"/>'
        '<w:rPr><w:b/><w:sz w:val="28"/><w:szCs w:val="28"/></w:rPr></w:style>'
        '<w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/>'
        '<w:basedOn w:val="Normal"/><w:next w:val="Normal"/>'
        '<w:rPr><w:b/><w:sz w:val="24"/><w:szCs w:val="24"/></w:rPr></w:style>'
        "</w:styles>"
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        '<Override PartName="/word/styles.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>'
        "</Types>"
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="word/document.xml"/></Relationships>'
    )
    doc_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
        'Target="styles.xml"/></Relationships>'
    )
    with ZipFile(output_path, "w", ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", rels)
        zf.writestr("word/document.xml", document_xml)
        zf.writestr("word/styles.xml", styles_xml)
        zf.writestr("word/_rels/document.xml.rels", doc_rels)


def main() -> None:
    questions = read_docx_paragraphs(SOURCE_QUESTIONS_DOCX)

    burnin_metrics = pd.read_csv(ELO_OUTPUT_DIR / "elo_burnin_prediction_metrics.csv")
    burnin_stability = pd.read_csv(ELO_OUTPUT_DIR / "elo_burnin_vs_1985_reference.csv")
    rerun = pd.read_csv(ELO_OUTPUT_DIR / "elo_single_year_rerun_convergence_decisions.csv")
    match_vol = pd.read_csv(ELO_OUTPUT_DIR / "elo_event_level_volatility_match_summary.csv")
    event_vol = pd.read_csv(ELO_OUTPUT_DIR / "elo_event_level_volatility_event_summary.csv")
    event_size = pd.read_csv(ELO_OUTPUT_DIR / "elo_event_level_volatility_by_event_size.csv")
    candidates = pd.read_csv(ELO_OUTPUT_DIR / "elo_candidate_baselines.csv")

    key_starts = [1985, 2005, 2010, 2015, 2025]
    key_metrics = burnin_metrics[burnin_metrics["start_year"].isin(key_starts)].copy()
    metric_rows = []
    for row in key_metrics.itertuples(index=False):
        metric_rows.append(
            [
                row.setting_name,
                int(row.start_year),
                int(row.evaluation_games),
                fmt6(row.log_loss),
                fmt6(row.brier_score),
                fmt6(row.accuracy),
            ]
        )

    vb_active = burnin_stability[
        (burnin_stability["setting_name"] == "validation_best_k30_scale300")
        & (burnin_stability["player_subset"] == "active_2025_games_ge5")
        & (burnin_stability["comparison_start_year"].isin([2000, 2005, 2010, 2015, 2020, 2025]))
    ]
    stability_rows = [
        [
            int(row.comparison_start_year),
            int(row.number_of_common_players),
            fmt(row.spearman_rank_correlation, 4),
            fmt(row.mean_abs_rating_difference, 2),
            fmt(row.top50_overlap, 2),
            fmt(row.top100_overlap, 2),
        ]
        for row in vb_active.itertuples(index=False)
    ]

    vb_compare = burnin_stability[
        (burnin_stability["setting_name"] == "validation_best_k30_scale300")
        & (burnin_stability["player_subset"].isin(["all_common_players", "active_2025_games_ge5"]))
        & (burnin_stability["comparison_start_year"].isin([2005, 2010, 2015, 2020]))
    ]
    subset_rows = [
        [
            row.player_subset,
            int(row.comparison_start_year),
            int(row.number_of_common_players),
            fmt(row.spearman_rank_correlation, 4),
            fmt(row.mean_abs_rating_difference, 2),
            fmt(row.top50_overlap, 2),
            fmt(row.top100_overlap, 2),
        ]
        for row in vb_compare.itertuples(index=False)
    ]

    rerun_rows = [
        [
            int(row.year),
            row.setting_name,
            "是" if bool(row.converged) else "否",
            int(row.total_iterations_run),
            fmt(row.final_mean_abs_change, 3),
            fmt(row.final_max_abs_change, 3),
            fmt(row.final_spearman_rank_correlation, 6),
        ]
        for row in rerun.itertuples(index=False)
    ]

    match_vol_rows = [
        [
            row.setting_name,
            fmt(row.mean_abs_match_update, 3),
            fmt(row.p90_abs_match_update, 3),
            fmt(row.p95_abs_match_update, 3),
            fmt(row.max_abs_match_update, 3),
        ]
        for row in match_vol.itertuples(index=False)
    ]
    event_vol_rows = [
        [
            row.setting_name,
            fmt(row.mean_abs_event_net_change, 3),
            fmt(row.median_abs_event_net_change, 3),
            fmt(row.p90_abs_event_net_change, 3),
            fmt(row.mean_cumulative_abs_match_updates_in_event, 3),
            fmt(row.mean_event_cancellation_ratio, 3),
        ]
        for row in event_vol.itertuples(index=False)
    ]
    vb_event_size = event_size[event_size["setting_name"] == "validation_best_k30_scale300"]
    event_size_rows = [
        [
            row.games_in_event_bucket,
            int(row.player_event_records),
            fmt(row.mean_abs_event_net_change, 3),
            fmt(row.mean_cumulative_abs_match_updates, 3),
            fmt(row.mean_event_cancellation_ratio, 3),
        ]
        for row in vb_event_size.itertuples(index=False)
    ]

    candidate_rows = [
        [
            row.candidate_name,
            f"K={fmt(row.k, 0)}, scale={fmt(row.scale, 0)}",
            row.role,
            int(row.burnin_candidate_start_year),
            fmt6(row["2025_full_history_log_loss"]),
            fmt(row.mean_abs_match_update, 3),
            row.final_recommendation,
        ]
        for _, row in candidates.iterrows()
    ]

    doc: list[str] = []
    doc.append(paragraph("Elo 模型确认问题回答总结", "Title"))
    doc.append(paragraph(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}"))
    doc.append(paragraph("来源：问elo.docx，以及 outputs/elo_optimization/ 下已经生成的 compact CSV/Markdown。"))
    doc.append(paragraph("说明：本报告只复核已有 Elo 结果，没有重跑 Elo，没有读取 2.24GB 的 elo_burnin_update_history_all_runs.csv。"))

    doc.append(paragraph("一、原始确认问题概览", "Heading1"))
    for item in questions:
        doc.append(bullet(item))

    doc.append(paragraph("二、Burn-in rating list stability：回答确认问题", "Heading1"))
    doc.append(paragraph("问题 1：历史越长，2025 prediction metrics 是不是继续变好，还是到某个年份后变化很小？", "Heading2"))
    doc.append(paragraph("回答：总体上历史越长越好，1985 full-history 在三组 Elo setting 上都给出最低 log loss；但从 1985 到 2005 的差距相对较小，2025-only 明显变差。"))
    doc.append(
        table(
            ["setting", "start year", "2025 games", "log loss", "Brier", "accuracy"],
            metric_rows,
        )
    )
    doc.append(paragraph("解释：这支持使用长历史 burn-in。若需要较短、计算更方便的 burn-in，2005 在 active player ranking 稳定性上是一个可辩护的 empirical candidate，但 1985 仍是最稳妥 reference。"))

    doc.append(paragraph("问题 2：2000/2005/2010/2015/2020/2025 与 1985 full-history final rating list 差多少？", "Heading2"))
    doc.append(paragraph("回答：以 validation-best Elo（K=30, scale=300）和 active_2025_games_ge5 为重点，2005 和 2010 与 1985 reference 仍然高度相似；2015 后差异开始扩大；2025-only 明显不稳定。"))
    doc.append(
        table(
            ["comparison start", "common players", "Spearman", "mean abs rating diff", "Top50 overlap", "Top100 overlap"],
            stability_rows,
        )
    )

    doc.append(paragraph("问题 3：active 2025 players 是否比 all players 更稳定？", "Heading2"))
    doc.append(paragraph("回答：是。历史数据对全部历史选手、尤其是不活跃或早期选手影响更大；对当前活跃选手 ranking list 的影响较小。这个可以作为 meeting 中回应导师 burn-in 质疑的重点。"))
    doc.append(
        table(
            ["subset", "start year", "common players", "Spearman", "mean abs rating diff", "Top50 overlap", "Top100 overlap"],
            subset_rows,
        )
    )
    doc.append(paragraph("小结：1985 full-history 最适合作为 reference；2005 是 active 2025 players 上较合理的 shorter burn-in candidate，但这只是 empirical diagnostic，不是理论规则。"))

    doc.append(paragraph("三、Single-year repeated rerun convergence：回答确认问题", "Heading1"))
    doc.append(paragraph("问题：同一年重复喂给 Elo 后，rating list 是否稳定？不同 K 是否收敛速度不同？", "Heading2"))
    doc.append(paragraph("回答：9 个 year-setting combinations 在 50 次内都没有达到严格 numerical convergence threshold。Spearman rank correlation 非常高，说明 rank ordering almost stable；但 rating values 仍然持续变化，说明单年重复运行不能替代 historical burn-in。"))
    doc.append(
        table(
            ["year", "setting", "converged?", "iterations", "final mean abs change", "final max abs change", "final Spearman"],
            rerun_rows,
        )
    )
    doc.append(paragraph("解释：这一步不是 prediction evaluation，也不是为了选最终模型。它支持这样的说法：只用一年数据反复运行可以让排名顺序接近稳定，但 rating 数值仍未严格收敛，因此需要足够历史数据作为 burn-in。"))

    doc.append(paragraph("四、Event-level volatility：回答确认问题", "Heading1"))
    doc.append(paragraph("问题 1：K 越大，单场平均 update 和高分位 update 是否明显更大？", "Heading2"))
    doc.append(paragraph("回答：是。K/scale 越 aggressive，match-level update 明显增大。validation-best K=30 scale=300 的 mean abs match update 为 11.690，高于 default K=20 scale=500 的 8.200。"))
    doc.append(
        table(
            ["setting", "mean abs match update", "p90", "p95", "max"],
            match_vol_rows,
        )
    )

    doc.append(paragraph("问题 2：event 内的 net rating change 是否比逐场累计变化更平滑？", "Heading2"))
    doc.append(paragraph("回答：是。event-level net change 明显小于 cumulative abs match updates。以 validation-best 为例，mean cumulative event movement 约 59.578，但 mean net event change 约 23.860，mean cancellation ratio 约 0.528。"))
    doc.append(
        table(
            ["setting", "mean abs event net", "median abs event net", "p90 event net", "mean cumulative match movement", "mean cancellation ratio"],
            event_vol_rows,
        )
    )
    doc.append(paragraph("按 event size 看，赛事越长，抵消越明显。validation-best 下 10+ games 的 mean cancellation ratio 约 0.295，说明逐场波动在 event-level 视角下会被明显平滑。"))
    doc.append(
        table(
            ["games in event", "records", "mean event net", "mean cumulative movement", "mean cancellation ratio"],
            event_size_rows,
        )
    )

    doc.append(paragraph("五、对 Elo baseline 的最终确认回答", "Heading1"))
    doc.append(paragraph("基于 burn-in、single-year rerun 和 event-level volatility 三个 diagnostic，Elo baseline 不应该只选一个唯一模型，而应该保留三个有不同用途的 baseline："))
    doc.append(
        table(
            ["candidate", "parameters", "role", "burn-in candidate", "1985 2025 log loss", "mean match update", "recommendation"],
            candidate_rows,
        )
    )
    doc.append(paragraph("最终可对导师表述为：Default Elo 作为 transparent simple baseline；Validation-best Elo 作为 prediction-oriented baseline；Conservative Elo 作为 stability-oriented reference。Aggressive K=35/K=40 只作为 sensitivity/warning examples，不作为主 baseline。"))

    doc.append(paragraph("六、可以直接对导师说的简短结论", "Heading1"))
    doc.append(bullet("Burn-in：1985 full-history 是最稳妥 reference；2005 对 active 2025 players 来说是可辩护的 shorter burn-in candidate，但不是理论规则。"))
    doc.append(bullet("Rating list stability：active 2025 players 的 Top50/Top100 overlap 和 Spearman correlation 很高，说明当前活跃 ranking list 对早期历史截断不太敏感；但 2020/2025-only 明显不够稳定。"))
    doc.append(bullet("Single-year rerun：rank ordering almost stable，但 rating values 未在 50 次内严格收敛，支持 historical burn-in 的必要性。"))
    doc.append(bullet("Volatility：K 越 aggressive，match-level update 越大；但 event-level net change 明显小于逐场累计变化，可以回应导师关于 tournament/event resolution 的观点。"))
    doc.append(bullet("Baseline decision：后续和 Glicko fair comparison 时，应同时报告 prediction metrics、rating list stability 和 volatility，而不是只看 log loss。"))

    make_docx(doc, OUTPUT_DOCX)
    print(f"Created {OUTPUT_DOCX}")


if __name__ == "__main__":
    main()
