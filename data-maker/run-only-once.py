



import os
import re
import json
import math
import random
import string
from datetime import date, timedelta

import pandas as pd
from faker import Faker

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak
)
from reportlab.pdfgen import canvas

# ============================================================
# CONFIGURATION
# ============================================================
NUM_PERSONS = 100
FAKER_LOCALE = "en_IN"       # Change if you want another locale
OUTPUT_DIR = "../data-maker-output"
PDF_DIR = "../pdf-policy-documents"
# SAVED_PDF_DIR = "../saved-pdf-policy-documents"
# PDF_DIR = os.path.join(OUTPUT_DIR, "individual_pdfs")
JSON_FILE = os.path.join(OUTPUT_DIR, "insurance_data.json")
CSV_FILE = os.path.join(OUTPUT_DIR, "insurance_data.csv")
SUMMARY_PDF = os.path.join(OUTPUT_DIR, "insurance_summary.pdf")

random.seed(42)
fake = Faker(FAKER_LOCALE)
Faker.seed(42)

# ============================================================
# HELPERS
# ============================================================
def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PDF_DIR, exist_ok=True)

def safe_filename(text: str) -> str:
    text = re.sub(r"[^\w\s-]", "", text).strip()
    text = re.sub(r"[-\s]+", "_", text)
    return text[:80]

def generate_policy_id(existing_ids: set) -> str:
    """
    Create a unique 12-character alphanumeric Policy ID.
    Example: A7K91P2QX4MZ
    """
    chars = string.ascii_uppercase + string.digits
    while True:
        pid = "".join(random.choices(chars, k=12))
        if pid not in existing_ids:
            existing_ids.add(pid)
            return pid

def random_gender():
    return random.choice(["Male", "Female", "Other"])

def random_policy_type():
    return random.choice([
        "Health Insurance",
        "Term Life Insurance",
        "Personal Accident Insurance",
        "Family Floater Insurance",
        "Senior Citizen Health Insurance"
    ])

def calculate_end_date(start_date: date, years: int = 1) -> date:
    # Approximate 1-year/2-year policy terms safely
    return start_date + timedelta(days=365 * years)

def make_nominee_name():
    return fake.name()

def generate_person_record(existing_ids: set) -> dict:
    gender = random_gender()

    # Keep age realistic for an insurance demo
    dob = fake.date_of_birth(minimum_age=21, maximum_age=70)
    issue_date = fake.date_between(start_date="-2y", end_date="today")
    policy_years = random.choice([1, 1, 1, 2, 3])
    expiry_date = calculate_end_date(issue_date, years=policy_years)

    sum_assured = random.choice([200000, 300000, 500000, 750000, 1000000, 1500000, 2000000])
    annual_premium = round(sum_assured * random.uniform(0.012, 0.065), 2)

    full_name = fake.name()
    address = fake.address().replace("\n", ", ")

    return {
        "policy_id": generate_policy_id(existing_ids),
        "full_name": full_name,
        "gender": gender,
        "date_of_birth": dob.strftime("%Y-%m-%d"),
        "email": fake.email(),
        "phone": fake.phone_number(),
        "address": address,
        "city": fake.city(),
        "state": fake.state(),
        "postal_code": fake.postcode(),
        "nominee_name": make_nominee_name(),
        "relationship_to_nominee": random.choice(["Spouse", "Parent", "Child", "Sibling"]),
        "policy_type": random_policy_type(),
        "issue_date": issue_date.strftime("%Y-%m-%d"),
        "expiry_date": expiry_date.strftime("%Y-%m-%d"),
        "policy_term_years": policy_years,
        "sum_assured": sum_assured,
        "annual_premium": annual_premium,
        "insurer_name": "Sample Secure Life Insurance Co.",
        "document_note": "SYNTHETIC / SAMPLE DOCUMENT - FOR TESTING ONLY"
    }

# ============================================================
# PDF WATERMARK CALLBACK
# ============================================================
def add_watermark_and_footer(canv, doc):
    canv.saveState()

    # Watermark
    canv.setFont("Helvetica-Bold", 40)
    canv.setFillColorRGB(0.85, 0.85, 0.85)
    canv.translate(300, 420)
    canv.rotate(45)
    canv.drawCentredString(0, 0, "SYNTHETIC / SAMPLE")
    canv.rotate(-45)
    canv.translate(-300, -420)

    # Footer
    canv.setFont("Helvetica", 8)
    canv.setFillColor(colors.grey)
    canv.drawString(20 * mm, 10 * mm, "Generated for testing/training only")
    canv.drawRightString(190 * mm, 10 * mm, f"Page {doc.page}")

    canv.restoreState()

# ============================================================
# INDIVIDUAL PDF GENERATOR
# ============================================================
def create_individual_pdf(record: dict, output_path: str):
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleCustom",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.darkblue,
        alignment=TA_CENTER,
        spaceAfter=10
    )
    section_style = ParagraphStyle(
        "SectionCustom",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#0B5394"),
        spaceBefore=8,
        spaceAfter=6
    )
    normal = ParagraphStyle(
        "NormalCustom",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=12,
        alignment=TA_LEFT
    )

    story = []

    # Title block
    story.append(Paragraph("Personal Insurance Policy Document", title_style))
    story.append(Paragraph(record["insurer_name"], styles["Heading3"]))
    story.append(Spacer(1, 6))

    note = f"<b>{record['document_note']}</b>"
    story.append(Paragraph(note, ParagraphStyle(
        "note",
        parent=normal,
        textColor=colors.red,
        alignment=TA_CENTER,
        fontSize=10,
        leading=12
    )))
    story.append(Spacer(1, 10))

    # Overview table
    overview_data = [
        ["Policy ID", record["policy_id"], "Policy Type", record["policy_type"]],
        ["Issue Date", record["issue_date"], "Expiry Date", record["expiry_date"]],
        ["Policy Term", f"{record['policy_term_years']} year(s)", "Annual Premium", f"INR {record['annual_premium']:,.2f}"],
        ["Sum Assured", f"INR {record['sum_assured']:,.2f}", "Nominee", record["nominee_name"]],
    ]

    overview_table = Table(overview_data, colWidths=[28*mm, 58*mm, 28*mm, 58*mm])
    overview_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.whitesmoke),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTNAME", (0, 0), (-1, -1, ), "Helvetica"),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("PADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(overview_table)
    story.append(Spacer(1, 12))

    # Personal details
    story.append(Paragraph("Policyholder Details", section_style))
    personal_data = [
        ["Full Name", record["full_name"]],
        ["Gender", record["gender"]],
        ["Date of Birth", record["date_of_birth"]],
        ["Email", record["email"]],
        ["Phone", record["phone"]],
        ["Address", record["address"]],
        ["City / State / Postal Code", f"{record['city']}, {record['state']} - {record['postal_code']}"],
    ]
    personal_table = Table(personal_data, colWidths=[48*mm, 112*mm])
    personal_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#D9EAF7")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("PADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(personal_table)
    story.append(Spacer(1, 12))

    # Nominee details
    story.append(Paragraph("Nominee Details", section_style))
    nominee_text = (
        f"The nominee for this policy is <b>{record['nominee_name']}</b> "
        f"with relationship recorded as <b>{record['relationship_to_nominee']}</b>."
    )
    story.append(Paragraph(nominee_text, normal))
    story.append(Spacer(1, 12))

    # Terms
    story.append(Paragraph("Sample Terms", section_style))
    terms = [
        "This is a system-generated synthetic insurance document created only for demonstration, development, QA, or training purposes.",
        "No real underwriting, claim processing, premium collection, or risk evaluation is associated with this file.",
        "All names, addresses, contact details, and policy numbers in this document are synthetic examples."
    ]
    for i, item in enumerate(terms, start=1):
        story.append(Paragraph(f"{i}. {item}", normal))
        story.append(Spacer(1, 3))

    story.append(Spacer(1, 18))
    story.append(Paragraph("Authorised Signatory (Sample)", section_style))
    story.append(Paragraph("Digital issuance for internal mock/testing use only.", normal))

    doc.build(story, onFirstPage=add_watermark_and_footer, onLaterPages=add_watermark_and_footer)

# ============================================================
# SUMMARY PDF
# ============================================================
def create_summary_pdf(df: pd.DataFrame, output_path: str):
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=12 * mm,
        rightMargin=12 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "summary_title",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=16,
        leading=20,
        textColor=colors.darkblue,
        alignment=TA_CENTER,
        spaceAfter=8
    )

    body = []
    body.append(Paragraph("Insurance Policy Summary Report", title_style))
    body.append(Paragraph(
        f"Total synthetic records generated: <b>{len(df)}</b>",
        styles["BodyText"]
    ))
    body.append(Spacer(1, 10))

    # Keep selected columns for landscape-friendly width, but within A4 portrait by slicing
    summary_cols = ["policy_id", "full_name", "policy_type", "issue_date", "expiry_date", "annual_premium"]
    table_data = [summary_cols] + df[summary_cols].values.tolist()

    # Paginate in chunks
    rows_per_page = 35
    for start in range(0, len(table_data), rows_per_page):
        chunk = table_data[start:start + rows_per_page]
        table = Table(
            chunk,
            colWidths=[28*mm, 50*mm, 38*mm, 22*mm, 22*mm, 26*mm]
        )
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F4E78")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 7.5),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.lightgrey]),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("PADDING", (0, 0), (-1, -1), 3),
        ]))
        body.append(table)
        if start + rows_per_page < len(table_data):
            body.append(PageBreak())

    doc.build(body, onFirstPage=add_watermark_and_footer, onLaterPages=add_watermark_and_footer)

# ============================================================
# MAIN PIPELINE
# ============================================================
def main():
    ensure_dirs()

    existing_ids = set()
    records = [generate_person_record(existing_ids) for _ in range(NUM_PERSONS)]

    # Save JSON
    with open(JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=4, ensure_ascii=False)

    # Save CSV with pandas
    df = pd.DataFrame(records)
    df.to_csv(CSV_FILE, index=False, encoding="utf-8")

    # Individual PDFs
    for idx, record in enumerate(records, start=1):
        filename = f"{idx:03d}_{record['policy_id']}_{safe_filename(record['full_name'])}.pdf"
        output_path = os.path.join(PDF_DIR, filename)
        create_individual_pdf(record, output_path)

    # Summary PDF
    create_summary_pdf(df, SUMMARY_PDF)

    print("Done.")
    print(f"JSON saved to: {JSON_FILE}")
    print(f"CSV saved to:  {CSV_FILE}")
    print(f"Individual PDFs saved to: {PDF_DIR}")
    print(f"Summary PDF saved to: {SUMMARY_PDF}")
    print(f"Total records: {len(records)}")

if __name__ == "__main__":
    main()

