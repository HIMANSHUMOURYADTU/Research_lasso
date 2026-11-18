"""Build a print-ready research report as a PDF.

Run from this folder:
  python build_report.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    CondPageBreak,
    HRFlowable,
    Image as RLImage,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
OUT = ROOT / "Communities_and_Crime_Report.pdf"
NAVY = colors.Color(0.11, 0.18, 0.32)
RULE = colors.Color(0.75, 0.77, 0.80)
ZEBRA = colors.Color(0.95, 0.96, 0.97)
MUTED = colors.Color(0.28, 0.30, 0.33)

PAGE_W, PAGE_H = A4
CONTENT_W = PAGE_W - 1.5 * inch


def draw_pipeline(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 8.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 16.2)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    def box(x, y, w, h, text, face="#f7f8fa", edge="#1d4e89"):
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.08,rounding_size=0.12",
            facecolor=face,
            edgecolor=edge,
            linewidth=1.1,
        )
        ax.add_patch(patch)
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=9,
            color="#1c1c1c",
            linespacing=1.35,
        )

    def arrow(x, y0, y1):
        ax.add_patch(
            FancyArrowPatch(
                (x, y0),
                (x, y1),
                arrowstyle="-|>",
                mutation_scale=11,
                linewidth=1.0,
                color="#1d4e89",
            )
        )

    box(1.5, 14.55, 7.0, 1.15, "Communities and Crime\n1,994 communities, 128 attributes")
    arrow(5, 14.5, 13.85)
    box(1.5, 12.7, 7.0, 1.05, "Drop 5 identifiers\nstate, county, community, name, fold")
    arrow(5, 12.65, 12.0)
    box(1.5, 10.85, 7.0, 1.05, "Drop 22 police-survey fields\nmissing for 84 percent of communities")
    arrow(5, 10.8, 10.15)
    box(1.5, 9.05, 7.0, 1.0, "100 predictors + ViolentCrimesPerPop")
    arrow(5, 9.0, 8.35)
    box(0.7, 7.15, 3.6, 1.05, "Training set\n1,595 communities", face="#eef3f8")
    box(5.7, 7.15, 3.6, 1.05, "Test set\n399 communities", face="#f8f1e8")
    ax.plot([3.2, 5], [8.2, 8.7], color="#1d4e89", linewidth=1.0)
    ax.plot([6.8, 5], [8.2, 8.7], color="#1d4e89", linewidth=1.0)
    arrow(5, 7.1, 6.45)
    box(1.3, 5.3, 7.4, 1.05, "Median impute and standardize\nfit on the training fold only")
    specs = [
        (0.25, "Least squares\nno penalty", "#f8ecec", "#8a3b3b"),
        (2.65, "Ridge\nL2 penalty", "#f8f1e8", "#8a5a12"),
        (5.05, "Lasso\nL1 penalty", "#eef6f0", "#2f7d4a"),
        (7.45, "Elastic net\nL1 + L2", "#f3eef8", "#5c3d86"),
    ]
    for x, text, face, edge in specs:
        box(x, 3.15, 2.3, 1.2, text, face=face, edge=edge)
        ax.annotate(
            "",
            xy=(x + 1.15, 4.4),
            xytext=(5.0, 5.25),
            arrowprops=dict(arrowstyle="-|>", color="#1d4e89", lw=0.9),
        )
    arrow(5, 3.1, 2.45)
    box(1.5, 1.15, 7.0, 1.15, "Compare on held-out communities\nR-squared, RMSE, MAE, coefficients")
    ax.text(
        5,
        0.45,
        "A separate gradient-descent run walks the same training loss from zero weights.",
        ha="center",
        va="center",
        fontsize=8,
        color="#3a3a3a",
    )
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def styles():
    base = dict(fontName="Times-Roman", fontSize=11, leading=15, textColor=colors.black, alignment=TA_JUSTIFY)
    return {
        "kicker": ParagraphStyle("kicker", fontName="Times-Bold", fontSize=9, leading=12, alignment=TA_CENTER, textColor=NAVY, tracking=1.2),
        "title": ParagraphStyle("title", fontName="Times-Bold", fontSize=20, leading=24, alignment=TA_CENTER, textColor=NAVY, spaceAfter=8),
        "subtitle": ParagraphStyle("subtitle", fontName="Times-Italic", fontSize=12, leading=16, alignment=TA_CENTER, textColor=MUTED),
        "meta": ParagraphStyle("meta", fontName="Times-Roman", fontSize=11, leading=15, alignment=TA_CENTER, textColor=colors.black),
        "h1": ParagraphStyle("h1", fontName="Times-Bold", fontSize=14, leading=18, textColor=NAVY, spaceBefore=12, spaceAfter=4, alignment=TA_LEFT),
        "h2": ParagraphStyle("h2", fontName="Times-Bold", fontSize=12, leading=15, textColor=colors.black, spaceBefore=10, spaceAfter=3, alignment=TA_LEFT),
        "body": ParagraphStyle("body", **base, spaceAfter=8),
        "caption": ParagraphStyle("caption", fontName="Times-Italic", fontSize=9, leading=12, textColor=MUTED, alignment=TA_LEFT, spaceBefore=2, spaceAfter=8),
        "center": ParagraphStyle("center", fontName="Times-Italic", fontSize=10.5, leading=14, alignment=TA_CENTER, textColor=colors.black, spaceBefore=2, spaceAfter=8),
        "ref": ParagraphStyle("ref", fontName="Times-Roman", fontSize=10, leading=13, leftIndent=14, firstLineIndent=-14, spaceAfter=6, alignment=TA_LEFT),
        "toc": ParagraphStyle("toc", fontName="Times-Roman", fontSize=11, leading=16, alignment=TA_LEFT),
    }


S = styles()
FIG_NO = 0


def P(text, style="body"):
    return Paragraph(text, S[style])


def rule():
    return HRFlowable(width="100%", thickness=0.6, color=NAVY, spaceBefore=0, spaceAfter=8)


def bullets(items):
    flow = []
    for item in items:
        flow.append(Paragraph(item, ParagraphStyle("li", parent=S["body"], leftIndent=14, bulletIndent=0, spaceAfter=3)))
    return flow


def scaled_image(path: Path, max_w: float, max_h: float) -> RLImage:
    from PIL import Image as PILImage

    with PILImage.open(path) as im:
        w, h = im.size
    disp_w = max_w
    disp_h = disp_w * (h / w)
    if disp_h > max_h:
        disp_h = max_h
        disp_w = disp_h * (w / h)
    img = RLImage(str(path), width=disp_w, height=disp_h)
    img.hAlign = "CENTER"
    return img


def add_figure(story, path: Path, caption: str, max_h=4.85 * inch):
    global FIG_NO
    FIG_NO += 1
    img = scaled_image(path, CONTENT_W, max_h)
    cap = Paragraph(f"<b>Figure {FIG_NO}.</b> {caption}", S["caption"])
    story.append(KeepTogether([img, Spacer(1, 3), cap]))


def nice_table(headers, rows, col_widths):
    data = [headers] + rows
    wrapped = []
    for r_i, row in enumerate(data):
        style = "body"
        new = []
        for c in row:
            if r_i == 0:
                new.append(Paragraph(f"<font color='white'><b>{c}</b></font>", ParagraphStyle("th", fontName="Times-Bold", fontSize=8, leading=10, alignment=TA_CENTER, textColor=colors.white)))
            else:
                align = TA_LEFT if False else TA_CENTER
                new.append(Paragraph(str(c), ParagraphStyle("td", fontName="Times-Roman", fontSize=8.5, leading=11, alignment=TA_CENTER)))
        # first column left aligned for readability
        if r_i > 0:
            new[0] = Paragraph(str(row[0]), ParagraphStyle("tdl", fontName="Times-Roman", fontSize=8.5, leading=11, alignment=TA_LEFT))
        wrapped.append(new)
    table = Table(wrapped, colWidths=col_widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
                ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("GRID", (0, 0), (-1, -1), 0.3, RULE),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, ZEBRA]),
            ]
        )
    )
    table.hAlign = "CENTER"
    return table


def fmt_coef(value: float) -> str:
    if abs(value) < 5e-4:
        return "0"
    return f"{value:+.3f}"


def first_page(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(NAVY)
    canvas.rect(0, PAGE_H - 10, PAGE_W, 10, fill=1, stroke=0)
    canvas.rect(0, 0, PAGE_W, 10, fill=1, stroke=0)
    canvas.setFillColor(MUTED)
    canvas.setFont("Times-Roman", 9)
    canvas.drawCentredString(PAGE_W / 2, 0.38 * inch, "1")
    canvas.restoreState()


def later_page(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(NAVY)
    canvas.setLineWidth(0.6)
    canvas.line(0.75 * inch, PAGE_H - 0.48 * inch, PAGE_W - 0.75 * inch, PAGE_H - 0.48 * inch)
    canvas.setFillColor(MUTED)
    canvas.setFont("Times-Italic", 8)
    canvas.drawString(0.75 * inch, PAGE_H - 0.40 * inch, "Predicting community violent-crime rates")
    canvas.drawRightString(PAGE_W - 0.75 * inch, PAGE_H - 0.40 * inch, "Himanshu")
    canvas.line(0.75 * inch, 0.48 * inch, PAGE_W - 0.75 * inch, 0.48 * inch)
    canvas.setFillColor(colors.black)
    canvas.setFont("Times-Roman", 9)
    canvas.drawCentredString(PAGE_W / 2, 0.28 * inch, str(doc.page))
    canvas.restoreState()


def build():
    results = json.loads((ROOT / "analysis_results.json").read_text(encoding="utf-8"))
    comparison = json.loads((FIG / "comparison.json").read_text(encoding="utf-8"))
    models = results["models"]
    base = results["baseline"]
    enet = models["ElasticNet"]
    labels = {
        "Linear": "least squares",
        "Ridge": "ridge",
        "Lasso": "the lasso",
        "ElasticNet": "elastic net",
    }
    short = {
        "Linear": "Least squares",
        "Ridge": "Ridge",
        "Lasso": "Lasso",
        "ElasticNet": "Elastic net",
    }
    fitted_names = ["Linear", "Ridge", "Lasso", "ElasticNet"]
    ranked = sorted(fitted_names, key=lambda n: models[n]["test"]["r2"], reverse=True)
    rank_text = ", then ".join(
        f"{labels[n]} ({models[n]['test']['r2']:.4f})" for n in ranked
    )
    holdout_gap = models[ranked[0]]["test"]["r2"] - models[ranked[-1]]["test"]["r2"]
    pipeline = FIG / "00_pipeline.png"
    draw_pipeline(pipeline)

    story = []
    story.append(Spacer(1, 1.15 * inch))
    story.append(P("TECHNICAL REPORT", "kicker"))
    story.append(Spacer(1, 10))
    story.append(P("Predicting Community Violent-Crime Rates", "title"))
    story.append(P("Least squares, ridge, the lasso, and elastic net<br/>on the UCI Communities and Crime data", "subtitle"))
    story.append(Spacer(1, 16))
    story.append(HRFlowable(width="40%", thickness=0.8, color=NAVY, spaceBefore=2, spaceAfter=14, hAlign="CENTER"))
    story.append(P("Himanshu", "meta"))
    story.append(Spacer(1, 4))
    story.append(P("Student technical report", "meta"))
    story.append(Spacer(1, 4))
    story.append(P("15 November 2026", "meta"))
    story.append(Spacer(1, 22))
    story.append(P("Dataset. UCI Machine Learning Repository, Communities and Crime.<br/>1,994 U.S. communities. 1990 Census, 1990 LEMAS survey, 1995 FBI Uniform Crime Reports.", "center"))
    story.append(PageBreak())

    story.append(P("1. Abstract", "h1"))
    story.append(rule())
    story.append(P(
        "This report asks whether community census characteristics predict the violent-crime rate, "
        "and whether ridge regression, the lasso, or elastic net is more accurate than ordinary least squares. "
        "The data are 1,994 U.S. communities from the UCI Communities and Crime collection. "
        "The target, ViolentCrimesPerPop, is already scaled to the unit interval. "
        "Identifiers are removed, and 22 police-survey fields that are missing for 84 percent of communities are dropped, leaving 100 predictors."
    ))
    story.append(P(
        "On a held-out fifth of the communities, least squares, ridge, the lasso, and elastic net all cut root-mean-square error "
        f"from {base['rmse']:.3f}, the error of predicting the training mean, to about {models[ranked[0]]['test']['rmse']:.3f}. "
        f"Test R-squared runs from {models[ranked[-1]]['test']['r2']:.3f} to {models[ranked[0]]['test']['r2']:.3f}. "
        "The gap between the four fits is much smaller than the spread from one train-test split to another. "
        "Elastic net, ridge, and the lasso are therefore not a meaningful accuracy upgrade on least squares. "
        "They are the better description of this file, because 49 predictor pairs correlate at 0.90 or above and least-squares weights on those copies are large and unstable. "
        f"Elastic net keeps {enet['n_nonzero']} of the 100 features on the primary split, with mixing weight l1 ratio {enet['l1_ratio']} and penalty alpha {enet['alpha']:.4g}. "
        "The associations are not causal effects."
    ))

    story.append(P("2. Introduction", "h1"))
    story.append(rule())
    story.append(P(
        "The practical question is whether a linear model can rank communities by violent crime using information that is known from the census, "
        "before looking at the crime counts themselves. The scientific question for the modelling is narrower. "
        "The same design matrix is given to four estimators that differ in how they treat collinear predictors: "
        "ordinary least squares, ridge regression (Hoerl and Kennard, 1970), the lasso (Tibshirani, 1996), "
        "and the elastic net, which mixes the two penalties (Zou and Hastie, 2005). "
        "If the columns were nearly orthogonal, the four fits would agree. They do not, and the disagreement is the result worth reporting."
    ))
    story.append(P(
        "The report proceeds as follows. Section 3 describes the file and the target. Section 4 states the preprocessing, the four estimators, "
        "the holdout protocol, and the gradient-descent check. Section 5 is the exploratory picture. Section 6 gives the predictive results "
        "and the coefficient comparison. Section 7 answers whether the penalties help. Section 8 shows gradient descent, including a one-feature animation "
        "of the fitted line. Section 9 states the limits of the evidence, and Section 10 concludes."
    ))

    story.append(P("3. Data", "h1"))
    story.append(rule())
    story.append(P("3.1 Sources and shape", "h2"))
    story.append(P(
        "The file communities.data contains 1,994 rows and 128 comma-separated attributes. "
        "A question mark denotes a missing value. communities.names is the UCI attribute dictionary. "
        "The donor assembled the table from the 1990 U.S. Census Summary Tape Files, the 1990 Law Enforcement Management and Administrative Statistics (LEMAS) survey, "
        "and the 1995 FBI Uniform Crime Reports (Redmond and Baveja, 2002; UCI Machine Learning Repository, 2009)."
    ))
    story.append(P(
        "Every numeric attribute in this release was normalized to roughly the interval from 0 to 1 by an equal-interval scaling that preserves the shape of each column. "
        "Values more than three standard deviations from the mean were clipped to 0 or to 1. "
        "The scaling does not preserve ratios between columns, so a value of 0.40 on one feature is not comparable to 0.40 on another. "
        "All errors in this report are therefore in normalized crime-rate units, not in crimes per 100,000 people. "
        "The original per-capita scale cannot be recovered from this file."
    ))
    story.append(P("3.2 Target", "h2"))
    story.append(P(
        "The attribute to be predicted is ViolentCrimesPerPop, the sum of murder, rape, robbery, and assault, divided by population. "
        "Communities with inconsistent rape counts were omitted by the donor, and many of those omissions were in the Midwest. "
        "On the 1,994 communities that remain, the target has mean 0.238, median 0.150, standard deviation 0.233, and skewness 1.52. "
        "About 14 percent of communities sit above 0.5, and the upper tail includes the clipped value 1. "
        "The distribution is piled up at low values. A model that always predicts the mean will look accurate on the many quiet communities and will miss the right tail."
    ))
    story.append(CondPageBreak(5.2 * inch))
    add_figure(
        story,
        FIG / "01_target_distribution.png",
        "Distribution of the normalized violent-crime rate across all 1,994 communities. The mean is 0.24 and the median is 0.15.",
    )
    story.append(P("3.3 What was not used as a predictor", "h2"))
    story.append(P(
        "Five fields are marked non-predictive in the dictionary. state is a numeric code for a nominal category, so treating it as a magnitude would be meaningless. "
        "county and community are identifiers, missing for 59 percent and 59 percent of rows. communityname is a label. "
        "fold is a tag for a pre-assigned cross-validation split and is not a property of the community. All five were dropped."
    ))
    story.append(P(
        "A further 22 fields come from the LEMAS police survey: sworn officers, budgets, cars, drug units, and the racial composition of the department. "
        "Each is missing for 1,675 communities, which is 84 percent of the file. The survey covered departments with at least 100 officers plus a sample of smaller departments. "
        "Imputing those columns would invent a police department for most towns, so the columns were dropped rather than filled. "
        "LemasPctOfficDrugUn has no missing values in this release and was kept. One cell of OtherPerCap is missing. It is the only gap left in the modelling matrix, and it is filled with the training-set median inside the pipeline."
    ))
    story.append(CondPageBreak(4.4 * inch))
    add_figure(
        story,
        FIG / "02_missing_values.png",
        "Missingness that determined the design matrix. The 22 LEMAS fields are each 84 percent missing. County and community identifiers are also mostly missing and were not used as predictors.",
    )
    story.append(P(
        "The modelling matrix therefore has 100 predictors and one complete target. That is the design used for every result below."
    ))

    story.append(P("4. Methods", "h1"))
    story.append(rule())
    story.append(P("4.1 Pipeline", "h2"))
    story.append(P(
        "Figure 3 is the whole analysis. After the drops above, rows are split 80/20 into 1,595 training communities and 399 test communities, with random seed 42. "
        "The median imputer and the standardizer are fit on the training fold and then applied to the test fold. "
        "Standardizing after the split matters for the penalized models, because each penalty treats a unit change in every coefficient as equally expensive. "
        "The census columns are already on a 0-1 scale, but their variances still differ, so an extra standardization keeps the penalty fair."
    ))
    story.append(CondPageBreak(7.2 * inch))
    add_figure(
        story,
        pipeline,
        "Analysis pipeline from the raw UCI file to the four held-out comparisons. Gradient descent, discussed in Section 8, uses the same training matrix.",
        max_h=6.5 * inch,
    )
    story.append(P("4.2 The four estimators", "h2"))
    story.append(P(
        "Let X be the standardized training matrix, with a column of ones handled as an unpenalized intercept, and let y be the normalized crime rate. "
        "Ordinary least squares chooses the intercept and the weights to minimize the residual sum of squares. There is no penalty. "
        "With 100 correlated columns the solution exists, but it is not unique in any practical sense: many different weight vectors fit the training data almost equally well."
    ))
    story.append(P(
        "Ridge regression, as fit by scikit-learn, minimizes the residual sum of squares plus alpha times the sum of squared weights. The intercept is not penalized. "
        "The penalty pulls every weight toward zero and is the classical remedy for a collinear design (Hoerl and Kennard, 1970). "
        f"Alpha was chosen by 5-fold cross-validation on the training set, over 40 values spaced logarithmically from 0.0001 to 100. The selected value is {models['Ridge']['alpha']:.4g}."
    ))
    story.append(P(
        "The lasso minimizes a scaled residual sum of squares plus alpha times the sum of absolute weights (Tibshirani, 1996). "
        "The absolute-value penalty can set weights to exactly zero, so the lasso is also a feature-selection method. "
        "Alpha was chosen on the same training folds and the same 40-point grid. "
        f"The selected value is {models['Lasso']['alpha']:.4g}, and {models['Lasso']['n_nonzero']} of the 100 weights are nonzero. "
        "Ridge alpha and lasso alpha are not on the same numerical scale, because the two objectives scale the residual sum of squares differently. Each was chosen to minimize cross-validated training error for its own objective."
    ))
    story.append(P(
        "Elastic net uses both penalties at once (Zou and Hastie, 2005). In scikit-learn the objective is a scaled residual sum of squares, "
        "plus alpha times l1 ratio times the sum of absolute weights, plus 0.5 times alpha times (1 minus l1 ratio) times the sum of squared weights. "
        "The l1 ratio chooses the mix: 1 would be the lasso, and a value near 0 is mostly ridge. "
        f"Cross-validation searched l1 ratios 0.1, 0.5, 0.7, 0.9, and 0.95, and the same 40 alpha values. "
        f"It selected l1 ratio {enet['l1_ratio']} and alpha {enet['alpha']:.4g}, leaving {enet['n_nonzero']} nonzero weights. "
        + (
            "That ratio is close to ridge: most of the penalty is on squared weights, and only a minority of columns are set to zero. "
            if enet["l1_ratio"] <= 0.2
            else (
                "That ratio is close to the lasso, with a little ridge shrinkage on the weights it keeps. "
                if enet["l1_ratio"] >= 0.85
                else "That ratio is a genuine mix of the two penalties. "
            )
        )
        + "Alpha for elastic net is not on the same scale as ridge alpha or lasso alpha."
    ))
    story.append(P("4.3 How the models are scored", "h2"))
    story.append(P(
        "The test set is used once, after the penalty has been chosen. Four numbers are reported. "
        "R-squared is the fraction of test variance explained relative to the test mean of the target; it can be negative if the model is worse than predicting the mean. "
        "Root-mean-square error (RMSE) and mean absolute error (MAE) are in normalized crime-rate units. "
        "Adjusted R-squared on the test set reduces R-squared to charge the model for the number of features. "
        "With 399 test rows and 100 features that charge is severe, so adjusted R-squared is used only as a rough complexity check, not as the primary score."
    ))
    seed42 = next(row for row in comparison["repeated_splits"]["splits"] if row["seed"] == 42)
    story.append(P(
        "A mean baseline predicts the training-set average for every test community. Any useful model has to beat that line. "
        "Because a single split can flatter one estimator, the comparison is repeated on five seeds, 0, 1, 7, 21, and 42. "
        "On those repeats the penalty is retuned with a coarser 16-point grid, and elastic net still searches the same five l1 ratios. That check is about ranking, not about reproducing the exact primary alpha. "
        f"On seed 42 the coarser lasso selects alpha {seed42['Lasso']['alpha']:.4g} and {seed42['Lasso']['nonzero']} nonzero weights, "
        f"against {models['Lasso']['alpha']:.4g} and {models['Lasso']['n_nonzero']} on the fine grid. "
        f"Test R-squared is {seed42['Lasso']['r2']:.3f} on the coarse grid and {models['Lasso']['test']['r2']:.3f} on the fine grid. "
        f"The coarser elastic net selects l1 ratio {seed42['ElasticNet']['l1_ratio']} and alpha {seed42['ElasticNet']['alpha']:.4g}, "
        f"keeping {seed42['ElasticNet']['nonzero']} weights, against l1 ratio {enet['l1_ratio']}, alpha {enet['alpha']:.4g}, and {enet['n_nonzero']} weights on the fine grid. "
        f"Its test R-squared is {seed42['ElasticNet']['r2']:.3f} against {models['ElasticNet']['test']['r2']:.3f}."
    ))
    story.append(P("4.4 Gradient descent", "h2"))
    story.append(P(
        "Least squares in scikit-learn is a direct solve. To show the fit being adjusted, the same training mean squared error was also minimized by batch gradient descent, starting from an intercept of zero and weights of zero. "
        "The gradient of the mean squared error with respect to the weights is (2/n) times X transpose times the residual vector. "
        "For this quadratic, the largest stable step is the reciprocal of the largest eigenvalue of (1/n) Z transpose Z, where Z is the training matrix with a column of ones. That limit is 0.040. "
        "Three steps were run: 0.006, 0.032, and 0.054. The middle value is the one reported in detail."
    ))
    story.append(P(
        "A 100-feature fit is a hyperplane and cannot be drawn as a line. Section 8 therefore also fits a single feature, PctKids2Par, by the same rule, and records the moving line. "
        "That animation is a picture of the algorithm. It is not the 100-feature model."
    ))

    story.append(P("5. Exploratory analysis", "h1"))
    story.append(rule())
    story.append(P(
        "Before any model is fit, the question is which columns move with the crime rate, and which columns are copies of each other. "
        "Correlations below use all 1,994 communities after a median fill of the single missing OtherPerCap cell. That fill is only for description. The models impute inside the training fold."
    ))
    story.append(P("5.1 Associations with the crime rate", "h2"))
    story.append(P(
        "The strongest linear associations are family-structure measures, and they come in pairs of opposite sign. "
        "The share of children living with two parents (PctKids2Par) correlates at -0.74 with the crime rate. "
        "The share of children born to never-married parents (PctIlleg) correlates at +0.74. "
        "The share of two-parent families, the white population share, investment income, public assistance, divorce, and the Black population share follow, with absolute correlations from about 0.56 to 0.71. "
        "These are not ten separate explanations. The heatmap shows them moving together."
    ))
    top = results["top_correlations"][:10]
    corr_rows = [[row["feature"], f"{row['correlation']:+.2f}"] for row in top]
    story.append(nice_table(
        ["Feature", "Correlation with crime rate"],
        corr_rows,
        [3.6 * inch, 2.6 * inch],
    ))
    story.append(Spacer(1, 6))
    story.append(P("Table 1. Ten strongest Pearson correlations with ViolentCrimesPerPop. Negative values mean higher feature values accompany lower crime rates in this file.", "caption"))
    story.append(CondPageBreak(5.4 * inch))
    add_figure(
        story,
        FIG / "03_correlation_with_target.png",
        "Signed correlations with the normalized crime rate. Blue bars are negative associations and red bars are positive associations.",
        max_h=5.3 * inch,
    )
    story.append(CondPageBreak(5.6 * inch))
    add_figure(
        story,
        FIG / "04_correlation_heatmap.png",
        "Correlations among the twelve strongest predictors and the target. Dark red is a strong positive correlation and dark blue is a strong negative correlation. Family-structure columns form a block.",
        max_h=5.5 * inch,
    )
    story.append(CondPageBreak(5.2 * inch))
    add_figure(
        story,
        FIG / "05_feature_scatters.png",
        "The six strongest predictors against the crime rate, all 1,994 communities. The clouds slope as the correlations say, and they are wide: no single feature determines the rate.",
        max_h=4.7 * inch,
    )
    story.append(P("5.2 Collinearity", "h2"))
    story.append(P(
        "Collinearity is the reason a penalty might matter. Forty-nine predictor pairs have absolute correlation of at least 0.90. "
        "The closest pairs are almost the same measurement taken over a slightly different window or a slightly different definition: "
        "recent-immigrant shares at different horizons, owner-occupied house values at the lower quartile, median, and upper quartile, population and the urban population count, and female divorce with total divorce."
    ))
    pair_rows = [
        [p["a"], p["b"], f"{p['abs_correlation']:.3f}"]
        for p in results["collinear_pairs"][:8]
    ]
    story.append(nice_table(
        ["Feature A", "Feature B", "Absolute correlation"],
        pair_rows,
        [2.15 * inch, 2.15 * inch, 1.9 * inch],
    ))
    story.append(Spacer(1, 6))
    story.append(P("Table 2. Closest predictor pairs. Several exceed 0.99 and will be treated by least squares as interchangeable.", "caption"))
    story.append(P(
        f"Variance-inflation factors make the same point one column at a time. "
        f"Of the 100 predictors, {results['vif']['n_vif_over_10']} have a VIF above 10 and {results['vif']['n_vif_over_30']} have a VIF above 30. "
        f"The median VIF is {results['vif']['median_vif']}. "
        f"TotalPctDiv is a near-exact linear combination of the other divorce measures, so its VIF is infinite. "
        f"A coefficient on that column, holding the other divorce columns fixed, is not a quantity one can interpret on its own."
    ))

    story.append(P("6. Predictive results", "h1"))
    story.append(rule())
    story.append(P("6.1 Primary holdout", "h2"))
    story.append(P(
        "Table 3 is the primary comparison. The split is seed 42. Penalties were tuned only on the training communities."
    ))
    def penalty_text(name: str) -> str:
        if name == "Linear":
            return "none"
        model = models[name]
        if name == "ElasticNet":
            return f"alpha {model['alpha']:.4g}; l1 ratio {model['l1_ratio']}"
        return f"alpha {model['alpha']:.4g}"

    metric_rows = [
        ["Mean baseline", "none", "0", "—", f"{base['r2']:.3f}", f"{base['rmse']:.3f}", f"{base['mae']:.3f}"],
    ]
    for name in fitted_names:
        metric_rows.append([
            short[name],
            penalty_text(name),
            str(models[name]["n_nonzero"]),
            f"{models[name]['train']['r2']:.3f}",
            f"{models[name]['test']['r2']:.3f}",
            f"{models[name]['test']['rmse']:.3f}",
            f"{models[name]['test']['mae']:.3f}",
        ])
    story.append(nice_table(
        ["Model", "Penalty", "Features", "Train R2", "Test R2", "Test RMSE", "Test MAE"],
        metric_rows,
        [1.2*inch, 1.45*inch, 0.65*inch, 0.65*inch, 0.62*inch, 0.72*inch, 0.65*inch],
    ))
    story.append(Spacer(1, 6))
    story.append(P(
        "Table 3. Primary 80/20 split, seed 42, 399 test communities. R-squared, RMSE, and MAE are on the normalized crime rate. "
        "The lasso feature count is the number of nonzero standardized coefficients.",
        "caption",
    ))
    story.append(P(
        f"Relative to predicting the mean, RMSE falls from {base['rmse']:.3f} to about {models[ranked[0]]['test']['rmse']:.3f}. "
        f"On this split the order by test R-squared is {rank_text}. "
        f"The gap from first to last is {holdout_gap:.3f}. "
        f"Least squares fits the training data hardest (train R-squared {models['Linear']['train']['r2']:.3f}) and gives a little of that back on the test set. "
        f"Test adjusted R-squared is {models['Linear']['test']['adj_r2']:.3f} for least squares, "
        f"{models['Ridge']['test']['adj_r2']:.3f} for ridge, {models['Lasso']['test']['adj_r2']:.3f} for the lasso, "
        f"and {models['ElasticNet']['test']['adj_r2']:.3f} for elastic net, which is charged for {enet['n_nonzero']} columns rather than 100."
    ))
    story.append(CondPageBreak(4.2 * inch))
    add_figure(
        story,
        FIG / "06_holdout_metrics.png",
        "Holdout R-squared and RMSE for the four models on the 399 test communities. The vertical scale of R-squared runs from 0 to 1, which makes the tiny gap between the models obvious.",
    )
    story.append(P("6.2 Calibration and residuals", "h2"))
    story.append(P(
        "A high R-squared can still hide a bent fit. Figure 8 plots every test community. The red line is perfect prediction. "
        "All four clouds follow that line through the middle of the range and fan out at high crime rates. "
        "Communities whose true rate is near 1 are often predicted in the range 0.5 to 0.8. The models compress the right tail, which is what a squared-error fit does when the target is skewed and the signal is incomplete."
    ))
    story.append(CondPageBreak(4.3 * inch))
    add_figure(
        story,
        FIG / "07_predicted_vs_actual.png",
        "Predicted against actual normalized crime rate on the test set. Points on the red line would be exact. All four models under-predict the highest-crime communities.",
    )
    story.append(P(
        "The residual plots show the same fact another way. Residuals are actual minus predicted. They center on zero, so the models are not biased on average, "
        "but the spread grows once the predicted rate leaves the dense low-crime region. The residual histograms are roughly symmetric, with a longer positive tail: a few communities are much more violent than the linear combination of census features suggests. "
        "Nothing in the residual shape singles out one of the four estimators."
    ))
    story.append(CondPageBreak(5.2 * inch))
    add_figure(
        story,
        FIG / "08_residuals.png",
        "Test residuals against the prediction (top) and the residual histogram (bottom). The vertical spread widens as the predicted crime rate rises.",
        max_h=5.0 * inch,
    )
    story.append(P(
        "Binning the lasso predictions into deciles tells the same story in ten numbers. In the lowest decile the mean prediction is about 0.00 and the mean outcome is about 0.05. "
        "In the highest decile the mean prediction is 0.60 and the mean outcome is 0.62. The ranking is right. The level is slightly shrunk toward the mean, which is expected."
    ))

    story.append(P("6.3 Does the ranking survive another split?", "h2"))
    split_src = comparison["repeated_splits"]["splits"]
    summary = comparison["repeated_splits"]["summary"]
    lasso_nz = [row["Lasso"]["nonzero"] for row in split_src]
    enet_nz = [row["ElasticNet"]["nonzero"] for row in split_src]
    enet_l1_values = [row["ElasticNet"]["l1_ratio"] for row in split_src]
    enet_alphas = [row["ElasticNet"]["alpha"] for row in split_src]
    l1_counts = {}
    for value in enet_l1_values:
        l1_counts[value] = l1_counts.get(value, 0) + 1
    l1_bits = " and ".join(
        f"{value:g} on {count} seed" + ("s" if count != 1 else "")
        for value, count in sorted(l1_counts.items(), key=lambda item: (-item[1], item[0]))
    )
    story.append(P(
        "Table 4 repeats the entire fit on five seeds. The penalty is retuned inside each training set on a 16-point alpha grid, "
        "and elastic net also searches the same five l1 ratios as the primary run. "
        "Ridge selects alpha 46.4 on every seed. The lasso selects alpha 0.001 on every seed and keeps "
        f"between {min(lasso_nz)} and {max(lasso_nz)} features. "
        f"Elastic net selects l1 ratio {l1_bits}. Its alpha runs from {min(enet_alphas):.4g} to {max(enet_alphas):.4g}, and it keeps between {min(enet_nz)} and {max(enet_nz)} features. "
        "Those alphas differ from the primary run because the grid is coarser. The test scores on seed 42 do not."
    ))
    split_rows = []
    seed_winners = []
    for row in split_src:
        scores = {name: row[name]["r2"] for name in fitted_names}
        best = max(scores, key=scores.get)
        seed_winners.append((row["seed"], short[best], scores[best]))
        split_rows.append([
            str(row["seed"]),
            f"{row['Linear']['r2']:.4f}",
            f"{row['Ridge']['r2']:.4f}",
            f"{row['Lasso']['r2']:.4f}",
            f"{row['ElasticNet']['r2']:.4f}",
            short[best],
        ])
    mean_best = max(fitted_names, key=lambda n: summary[n]["mean_r2"])
    split_rows.append([
        "Mean",
        f"{summary['Linear']['mean_r2']:.4f}",
        f"{summary['Ridge']['mean_r2']:.4f}",
        f"{summary['Lasso']['mean_r2']:.4f}",
        f"{summary['ElasticNet']['mean_r2']:.4f}",
        short[mean_best],
    ])
    story.append(nice_table(
        ["Seed", "Linear", "Ridge", "Lasso", "Elastic net", "Highest"],
        split_rows,
        [0.62*inch, 0.95*inch, 0.85*inch, 0.85*inch, 1.15*inch, 1.35*inch],
    ))
    story.append(Spacer(1, 6))
    std_bits = ", ".join(
        f"{summary[n]['std_r2']:.3f} for {labels[n]}" for n in fitted_names
    )
    story.append(P(
        "Table 4. Test R-squared on five 80/20 splits. The mean is an unweighted average of the five seeds. "
        f"The standard deviation of test R-squared across seeds is {std_bits}.",
        "caption",
    ))
    winner_bits = "; ".join(
        f"seed {seed} goes to {name} ({score:.4f})" for seed, name, score in seed_winners
    )
    mean_bits = ", ".join(f"{short[n]} {summary[n]['mean_r2']:.4f}" for n in fitted_names)
    rmse_bits = ", ".join(f"{summary[n]['mean_rmse']:.4f} for {labels[n]}" for n in fitted_names)
    mean_gap = summary[mean_best]["mean_r2"] - min(summary[n]["mean_r2"] for n in fitted_names)
    win_bits = ", ".join(f"{short[n]} {summary[n]['wins']}" for n in fitted_names)
    story.append(P(
        f"{winner_bits[0].upper()}{winner_bits[1:]}. "
        f"Mean test R-squared is {mean_bits}. {short[mean_best]} has the highest average, by {mean_gap:.3f} over the lowest average. "
        f"Mean test RMSE is {rmse_bits}. "
        f"Wins across the five seeds: {win_bits}. "
        "A reader who looked only at one seed would name a winner. Averaging five splits still leaves a gap smaller than the noise of the split."
    ))

    story.append(P("7. Where the penalties are actually better", "h1"))
    story.append(rule())
    story.append(P(
        "Accuracy is a tie for any decision this data can support. The coefficients are not a tie. "
        "Table 5 lists the ten features on which least squares places the largest absolute weights, and the weights ridge, the lasso, and elastic net give those same features. "
        "Each number is the change in the normalized crime rate associated with a one-standard-deviation increase in that column, holding the other included columns fixed."
    ))
    comp = results["coefficient_compare"]

    def coef_of(model_name: str, feature: str) -> float:
        return comp[model_name][comp["features"].index(feature)]

    coef_rows = []
    for i, name in enumerate(comp["features"]):
        coef_rows.append([
            name,
            fmt_coef(comp["Linear"][i]),
            fmt_coef(comp["Ridge"][i]),
            fmt_coef(comp["Lasso"][i]),
            fmt_coef(comp["ElasticNet"][i]),
        ])
    coef_table = nice_table(
        ["Feature", "Least squares", "Ridge", "Lasso", "Elastic net"],
        coef_rows,
        [1.85*inch, 1.15*inch, 1.05*inch, 1.05*inch, 1.2*inch],
    )
    story.append(KeepTogether([
        coef_table,
        Spacer(1, 6),
        P(
            "Table 5. Standardized coefficients for the features with the largest least-squares weights. A zero means that penalty dropped the feature. "
            "Signs are partial associations, not effects of changing the feature.",
            "caption",
        ),
    ]))
    story.append(CondPageBreak(5.0 * inch))
    add_figure(
        story,
        FIG / "09_coefficients.png",
        "The same comparison as Table 5, drawn so the shrinkage is visible. Least-squares bars are several times longer than the penalized bars on the duplicated columns.",
        max_h=5.0 * inch,
    )
    story.append(P(
        "Two examples show why the long bars should not be believed. PctRecImmig8 and PctRecImmig10 correlate at 0.996. "
        "Least squares gives them +0.128 and -0.080. A community cannot have a high eight-year immigrant share and a low ten-year immigrant share; the two columns record almost the same fact. "
        "The opposite signs are the fit using one column to cancel the other. "
        f"Ridge moves them to {fmt_coef(coef_of('Ridge', 'PctRecImmig8'))} and {fmt_coef(coef_of('Ridge', 'PctRecImmig10'))}. "
        f"The lasso gives {fmt_coef(coef_of('Lasso', 'PctRecImmig8'))} and {fmt_coef(coef_of('Lasso', 'PctRecImmig10'))}. "
        f"Elastic net gives {fmt_coef(coef_of('ElasticNet', 'PctRecImmig8'))} and {fmt_coef(coef_of('ElasticNet', 'PctRecImmig10'))}."
    ))
    story.append(P(
        "TotalPctDiv correlates at +0.55 with the crime rate, in the same direction as FemalePctDiv and MalePctDivorce. "
        f"Its least-squares coefficient is {fmt_coef(coef_of('Linear', 'TotalPctDiv'))}, because the column is a copy of the other divorce measures and the VIF is infinite. "
        f"Ridge moves that partial coefficient to {fmt_coef(coef_of('Ridge', 'TotalPctDiv'))}. "
        f"The lasso gives {fmt_coef(coef_of('Lasso', 'TotalPctDiv'))}, and elastic net gives {fmt_coef(coef_of('ElasticNet', 'TotalPctDiv'))}. "
        "MalePctDivorce keeps a modest positive weight under the penalties, which matches the raw correlation."
    ))
    story.append(P(
        "The features that agree across the correlation table and the penalized models are the ones worth stating. "
        "A higher share of children in two-parent households and a higher share of households with investment income go with a lower predicted crime rate. "
        "A higher share of births to unmarried parents, a higher Black population share, and a higher male divorce rate go with a higher predicted rate. "
        "Even those statements are partial associations. Poverty, family structure, and racial composition are entangled in this file. "
        "PctPopUnderPov correlates at +0.52 with crime, but once the overlapping columns are in the model its lasso coefficient is slightly negative. "
        "That sign flip is shared variance. It is not evidence that poverty lowers crime."
    ))
    story.append(P(
        "The practical recommendation follows from the coefficient comparison, not from the third decimal of R-squared. "
        "Use least squares if the only product required is a prediction and the simplest program is the goal; on these splits it will not be meaningfully worse. "
        "Use ridge if every column should remain in the equation and the weights should stay small enough to read. "
        f"Use the lasso if the product is a shorter equation: on the primary split it keeps {models['Lasso']['n_nonzero']} columns, including by dropping exact copies, and matches the others on error. "
        f"Use elastic net when the product should be the mix cross-validation actually chose. On the primary split that mix is l1 ratio {enet['l1_ratio']}, alpha {enet['alpha']:.4g}, and {enet['n_nonzero']} nonzero weights, with test R-squared {models['ElasticNet']['test']['r2']:.3f}. "
        + (
            "That l1 ratio is close to the lasso, with a little extra ridge shrinkage on the weights it keeps. "
            if enet["l1_ratio"] >= 0.85
            else (
                "That l1 ratio is close to ridge, with only a little feature selection. "
                if enet["l1_ratio"] <= 0.2
                else "That l1 ratio is a genuine mix of the two penalties. "
            )
        )
        + "Do not describe any of the three penalties as a large reduction in prediction error relative to least squares."
    ))

    story.append(P("8. Gradient descent and the fitted line", "h1"))
    story.append(rule())
    story.append(P("8.1 Full model", "h2"))
    story.append(P(
        "Batch gradient descent on the 100 standardized training columns starts at zero weights and zero intercept. Training mean squared error starts at 0.115, which is the error of predicting zero for every community. "
        "The least-squares minimum on this matrix is 0.0167. With learning rate 0.032, the loss is 0.018 by step 50 and 0.0168 by step 1,000. "
        "At step 5,000 it is 0.0167, matching least squares to the fourth decimal. The holdout RMSE of that weight vector is 0.132, the same as the closed-form least-squares model."
    ))
    story.append(P(
        "The weights do not arrive when the loss does. The Euclidean distance from the gradient-descent weights to the least-squares weights starts at 0.45, is still 0.28 after 1,000 steps, and is 0.20 after 5,000 steps. "
        "This is the geometry of a collinear design. The loss is a long, flat valley. The error drops as soon as the fit moves onto the valley floor. "
        "Sliding along the floor toward one particular least-squares solution, among many that fit almost as well, takes far longer. "
        "Figure 11 shows how quickly the loss falls. Figure 12 shows that the weights are still moving after the loss has flattened."
    ))
    story.append(CondPageBreak(4.3 * inch))
    add_figure(
        story,
        FIG / "10_gradient_descent_loss.png",
        "Training mean squared error against the gradient step. Learning rate 0.032 reaches the least-squares loss faster than learning rate 0.006. Learning rate 0.054 diverges within about ten steps and is omitted because it leaves the axis.",
    )
    story.append(CondPageBreak(4.3 * inch))
    add_figure(
        story,
        FIG / "11_gradient_descent_weights_vs_loss.png",
        "Learning rate 0.032. The loss over the first 400 steps (left) falls to the least-squares value. The distance to the least-squares weight vector (right) is still declining at step 5,000.",
    )
    story.append(P(
        "Figure 13 draws every weight. The grey traces are the 100 coefficients. Four of the largest least-squares targets are highlighted, and the dotted horizontal lines are those targets. "
        "By step 5,000, PersPerOccupHous has climbed most of the way to its target. PctRecImmig8, TotalPctDiv, and PctPersOwnOccup are still far from the large least-squares values. "
        "That is consistent with Section 7: those large targets are the unstable directions, and gradient descent is slow exactly where the design is flat."
    ))
    story.append(CondPageBreak(4.6 * inch))
    add_figure(
        story,
        FIG / "12_gradient_descent_coefficient_paths.png",
        "All 100 standardized coefficients during gradient descent, starting at zero. Dotted lines are the least-squares values of the four highlighted features. Several of those targets have not been reached when the loss has already flattened.",
        max_h=4.6 * inch,
    )
    story.append(P("8.2 Two-feature pictures", "h2"))
    story.append(P(
        "A contour plot reduces the same geometry to two weights. The surface is training mean squared error. The star is the least-squares point for that pair alone, and the orange path is gradient descent from a poor start."
    ))
    story.append(P(
        "PctKids2Par and PctIlleg correlate at about -0.87. Their error surface is a valley rather than a round bowl. "
        "The path first moves across the steep wall, then travels along the valley, with the dots packing closer together as the slope flattens, and ends on the star. "
        "racepctblack and pctWInvInc correlate at 0.49. Their bowl is rounder, and the path reaches the star in fewer, more even steps. "
        "The full model is the first picture, repeated across dozens of overlapping pairs."
    ))
    story.append(CondPageBreak(5.2 * inch))
    add_figure(
        story,
        FIG / "13_gradient_descent_contour_family.png",
        "Gradient descent on PctKids2Par and PctIlleg only. The pale valley runs along the direction in which the two overlapping features trade off. The path slides down that valley into the least-squares point.",
        max_h=5.1 * inch,
    )
    story.append(CondPageBreak(5.2 * inch))
    add_figure(
        story,
        FIG / "14_gradient_descent_contour_bowl.png",
        "Gradient descent on racepctblack and pctWInvInc. The features overlap less (correlation 0.49), the bowl is rounder, and the path is shorter.",
        max_h=5.1 * inch,
    )
    story.append(P("8.3 The line fitting, one feature", "h2"))
    story.append(P(
        "Figures 16 to 18 are three frames of an animation stored as figures/15_line_fitting.gif. "
        "The points are the 1,595 training communities. The only predictor is PctKids2Par, on its original 0-1 scale, so the fitted object really is a line. "
        "The dashed grey line is the least-squares fit for this one feature: slope -0.84 and intercept 0.76. "
        "Gradient descent starts on purpose at the wrong tilt, slope +0.90 and intercept 0.05, which says that communities with more children in two-parent households would have more crime. The data say the opposite."
    ))
    story.append(P(
        "The learning rate is 0.40, just over half of the stable limit for this two-parameter problem. "
        "At step 0 the training mean squared error is 0.283. At step 39 the slope has crossed zero and is -0.25, and the error has fallen to about 0.04. "
        "At step 188 the slope is -0.83 and the intercept is 0.75, on top of the least-squares line, and the error is 0.025. "
        "The animation is the report's picture of adjustment. It is also a reminder of scale: this one feature, fully fit, has training mean squared error 0.025, while the 100-feature model reaches 0.017. "
        "Most of the linear signal that one strong column can carry is already in the scatter. The other 99 columns buy a smaller further reduction, and they buy the collinearity documented above."
    ))
    story.append(CondPageBreak(4.5 * inch))
    add_figure(
        story,
        FIG / "16_line_fit_start.png",
        "Step 0 of the one-feature fit. The orange line slopes the wrong way. The dashed line is the least-squares target. Training mean squared error is 0.283.",
    )
    story.append(CondPageBreak(4.5 * inch))
    add_figure(
        story,
        FIG / "17_line_fit_middle.png",
        "Step 39. The slope has become negative and the loss curve has dropped most of the way, but the line is still flatter than the least-squares fit.",
    )
    story.append(CondPageBreak(4.5 * inch))
    add_figure(
        story,
        FIG / "18_line_fit_end.png",
        "Step 188. The gradient-descent line lies on the least-squares line. Training mean squared error is 0.025, against a least-squares value of the same size for this single feature.",
    )

    story.append(P("9. Limits", "h1"))
    story.append(rule())
    story.append(P(
        "The target is a 1995 crime rate and the predictors are mostly 1990 census counts, both already clipped and scaled. "
        "Nothing here is a forecast for a later decade, a different country, or a raw count of crimes per 100,000. "
        "Communities that could not be matched across the census and the crime files, and communities with inconsistent rape figures, are simply absent."
    ))
    story.append(P(
        "The dropped LEMAS columns might carry information for the minority of cities that answered the survey. "
        "They were excluded because a model for all 1,994 communities cannot honestly use a column that is blank for five communities out of six. "
        "A separate study of the surveyed departments would be a different design, not a missing appendix to this one."
    ))
    story.append(P(
        "The linear model has no interactions and no nonlinearity beyond what is already baked into the normalized columns. "
        "The residual fan at high predicted crime is the visible cost of that choice. A more flexible model might reduce the tail error. "
        "It would not, by itself, turn a partial coefficient into a cause."
    ))
    story.append(P(
        "Racial composition, poverty, family structure, and housing appear together in the strongest correlations and in the penalized coefficients. "
        "A coefficient is the association left after the other included columns are held fixed in the equation. "
        "It is not the change in crime that would follow from changing that column in a city. "
        "The sign flip on the poverty share, once family structure is included, is the worked example."
    ))
    story.append(P(
        "Finally, the accuracy ranking is sensitive to the split at the third decimal and sensitive to the alpha grid at a level that does not change the test error in a material way. "
        "Claims sharper than \"the four linear models are tied on accuracy, and the penalized ones are more readable\" are not supported."
    ))

    story.append(P("10. Conclusion", "h1"))
    story.append(rule())
    story.append(P(
        "Census characteristics of a community carry a real linear signal about its normalized violent-crime rate. "
        "A model that uses them cuts holdout RMSE by about two-fifths relative to predicting the mean, and explains a bit over 60 percent of holdout variance. "
        "The signal is concentrated in overlapping blocks: two-parent households, births to unmarried parents, divorce, income, and racial composition."
    ))
    story.append(P(
        "Ordinary least squares, ridge, the lasso, and elastic net predict new communities equally well for any practical purpose. "
        f"Across five splits, mean test R-squared differs by {mean_gap:.3f}, an order of magnitude less than the variation between splits. "
        "The penalties earn their place in the report because the design is collinear. "
        "Least squares assigns large, cancelling weights to near-duplicate columns, including a negative weight on a divorce measure that is positively correlated with crime. "
        f"Ridge shrinks those weights. The lasso sets copies to zero and keeps {models['Lasso']['n_nonzero']} features on the primary split. "
        f"Elastic net, at l1 ratio {enet['l1_ratio']}, keeps {enet['n_nonzero']} features and shrinks the same unstable directions, with no meaningful loss of accuracy."
    ))
    story.append(P(
        "Gradient descent reaches the least-squares training loss and the least-squares holdout error, while many individual weights are still moving. "
        "That is the same collinearity, seen as a flat valley rather than as a variance-inflation factor. "
        "On the single strongest feature, the same algorithm can be watched directly: a line that starts sloping the wrong way rotates onto the downward least-squares fit within 188 steps."
    ))

    story.append(P("References", "h1"))
    story.append(rule())
    refs = [
        "Hoerl, A. E. and Kennard, R. W. (1970). Ridge regression: biased estimation for nonorthogonal problems. Technometrics, 12(1), 55-67.",
        "Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M. and Duchesnay, E. (2011). Scikit-learn: machine learning in Python. Journal of Machine Learning Research, 12, 2825-2830.",
        "Redmond, M. A. and Baveja, A. (2002). A data-driven software tool for enabling cooperative information sharing among police departments. European Journal of Operational Research, 141(3), 660-678.",
        "Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. Journal of the Royal Statistical Society, Series B, 58(1), 267-288.",
        "Zou, H. and Hastie, T. (2005). Regularization and variable selection via the elastic net. Journal of the Royal Statistical Society, Series B, 67(2), 301-320.",
        "UCI Machine Learning Repository (2009). Communities and Crime. Donor: M. Redmond. Compiled from the 1990 U.S. Census, the 1990 LEMAS survey, and the 1995 FBI Uniform Crime Reports.",
        "U.S. Department of Commerce, Bureau of the Census (1992). Census of Population and Housing, 1990: Summary Tape File 1A and Summary Tape File 3A.",
        "U.S. Department of Justice, Bureau of Justice Statistics (1992). Law Enforcement Management and Administrative Statistics, 1990.",
        "U.S. Department of Justice, Federal Bureau of Investigation (1995). Crime in the United States.",
    ]
    for ref in refs:
        story.append(Paragraph(ref, S["ref"]))

    story.append(Spacer(1, 8))
    story.append(P(
        "Reproduction. From the project folder, python crime_regression.py writes the primary metrics, "
        "python plot_analysis.py writes the still figures, python animate_fit.py writes the line-fitting animation and the five-split table, "
        "and python build_report.py writes this PDF. The primary split uses random seed 42 and a 40-point penalty grid. "
        "The five-split check uses seeds 0, 1, 7, 21, and 42 and a 16-point penalty grid.",
        "caption",
    ))

    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=A4,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.72 * inch,
        bottomMargin=0.62 * inch,
        title="Predicting Community Violent-Crime Rates",
        author="Himanshu",
        subject="Least squares, ridge, lasso, and elastic net on the UCI Communities and Crime data",
    )
    doc.build(story, onFirstPage=first_page, onLaterPages=later_page)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    build()
