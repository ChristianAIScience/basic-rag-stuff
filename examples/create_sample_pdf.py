"""Create the sample PDF data/handbook/employee_handbook.pdf.

Run:  uv run python -m examples.create_sample_pdf

The PDF uses three font sizes, which the PDF chunker turns into structure:
    18 pt  chapter   ("1 Welcome to Lumora")
    14 pt  section   ("1.1 Our History")
    11 pt  body text
Uses fpdf2 (a dev dependency) - only needed to generate the sample, not to read PDFs.
"""

from pathlib import Path

from fpdf import FPDF

OUTPUT = Path(__file__).resolve().parent.parent / "data" / "handbook" / "employee_handbook.pdf"

CHAPTER, SECTION, BODY = 18, 14, 11
PAGE_BREAK = None

CONTENT: list[tuple[int, str] | None] = [
    (BODY, "Lumora Systems Employee Handbook, edition 2026. This handbook applies to all employees "
           "at all Lumora sites."),
    (CHAPTER, "1 Welcome to Lumora"),
    (SECTION, "1.1 Our History"),
    (BODY, "Lumora Systems was founded in 2014 in Leipzig by Mara Feldmann and Jonas Ortiz. "
           "The first product, the Glowbeam Classic desk lamp, was released in 2016. "
           "Today Lumora employs 140 people at two sites: Leipzig (headquarters) and Lisbon."),
    (SECTION, "1.2 Our Values"),
    (BODY, "Our three values are Light, Clarity and Care. Light: we build products that make "
           "everyday life brighter. Clarity: we communicate openly and document decisions. "
           "Care: we look after our customers, our colleagues and the environment."),
    (CHAPTER, "2 Working at Lumora"),
    (SECTION, "2.1 Onboarding"),
    (BODY, "Every new employee is assigned an onboarding buddy for the first three months. "
           "Laptops are picked up on the first day in room L-204 at the Leipzig office. "
           "The first week includes a product workshop and a tour of the lamp assembly line."),
    (SECTION, "2.2 Business Travel"),
    (BODY, "Trips shorter than 600 kilometres must be made by train. First-class tickets are "
           "allowed for train journeys longer than four hours. Flights require approval from the "
           "department head and must be booked in economy class. The hotel limit is 140 euros per "
           "night in most cities and 190 euros per night in London, Paris and Zurich. Travel "
           "expenses are submitted in Ledgerly within 30 days after the trip. Per diem rates "
           "follow the official German tax tables. Employees may combine business trips with "
           "private vacation days, but any additional costs caused by the private part of the "
           "trip are paid by the employee. Rental cars are only allowed when no reasonable "
           "public transport connection exists, and electric cars must be preferred."),
    PAGE_BREAK,
    (CHAPTER, "3 Security, Safety and Protecting Company Information at All Sites"),
    (SECTION, "3.1 Visitors"),
    (BODY, "All visitors must register at reception and wear a yellow visitor badge. "
           "Visitors must be accompanied by a Lumora employee at all times, including "
           "in the cafeteria."),
    (SECTION, "3.2 Fire Safety"),
    (BODY, "The assembly point in Leipzig is parking lot P2. Fire drills take place twice a "
           "year, in March and September. Every floor has two trained fire wardens, who wear "
           "orange vests during drills."),
]


def main() -> None:
    pdf = FPDF(format="A4")
    pdf.add_page()
    for item in CONTENT:
        if item is PAGE_BREAK:
            pdf.add_page()
            continue
        size, text = item
        pdf.set_font("Helvetica", style="B" if size > BODY else "", size=size)
        if size > BODY:
            pdf.ln(4)  # some space above headings
        pdf.multi_cell(0, size * 0.55, text, new_x="LMARGIN", new_y="NEXT")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(OUTPUT))
    print(f"Created {OUTPUT}")


if __name__ == "__main__":
    main()
