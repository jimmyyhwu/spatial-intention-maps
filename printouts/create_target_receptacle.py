from fpdf import FPDF

receptacle_width = 150
line_width = 5
dash_length = 11.67
pdf_name = 'receptacle.pdf'

# Paper params
orientation = 'L'
width_in, height_in = 11, 8.5
mm_per_in = 25.4
width, height = width_in * mm_per_in, height_in * mm_per_in

# Create PDF
pdf = FPDF(orientation, 'mm', 'letter')
pdf.add_page()
pdf.set_line_width(line_width)
offset = receptacle_width / 2 - line_width / 2
for (x1, x2, y1, y2) in [(-1, 1, -1, -1), (1, 1, -1, 1), (1, -1, 1, 1), (-1, -1, 1, -1)]:
    pdf.dashed_line(
        width / 2 + x1 * offset, height / 2 + y1 * offset,
        width / 2 + x2 * offset, height / 2 + y2 * offset,
        dash_length=dash_length, space_length=dash_length + 2 * line_width
    )

# Save PDF
pdf.output(pdf_name)
