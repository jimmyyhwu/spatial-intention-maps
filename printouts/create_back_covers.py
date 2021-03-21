from fpdf import FPDF

template_width = 75
template_height = 47
pdf_name = 'back-covers.pdf'

# Paper params
orientation = 'P'
width_in, height_in, margin_in, side_margin_in = 8.5, 11, 0.5, 0.75
mm_per_in = 25.4
width, height, margin, side_margin = width_in * mm_per_in, height_in * mm_per_in, margin_in * mm_per_in, side_margin_in * mm_per_in

# Create PDF
pdf = FPDF(orientation, 'mm', 'letter')
pdf.add_page()
pdf.set_line_width(0.01)

def draw_template(x, y):
    pdf.rect(x, y, template_width, template_height)
    pdf.line(x, y + 28, x + template_width, y + 28)
    pdf.line(x + 15, y, x + 15, y + template_height)
    pdf.line(x + 60, y, x + 60, y + template_height)
    pdf.line(x + 15 + 45 / 2, y, x + 15 + 45 / 2, y + 3)
    pdf.line(x + 15 + (45 - 12) / 2, y + template_height, x + 15 + (45 - 12) / 2, y + template_height - 3)
    pdf.line(x + 60 - (45 - 12) / 2, y + template_height, x + 60 - (45 - 12) / 2, y + template_height - 3)

# Draw templates
num_rows, num_cols = 5, 2
spacing_y = (height - 2 * margin - num_rows * template_height) / (num_rows + 1)
spacing_x = (width - 2 * side_margin - num_cols * template_width) / (num_cols + 1)
for i in range(num_rows):
    corner_y = margin + spacing_y + i * (template_height + spacing_y)
    for j in range(num_cols):
        corner_x = side_margin + spacing_x + j * (template_width + spacing_x)
        draw_template(corner_x, corner_y)

# Save PDF
pdf.output(pdf_name)
