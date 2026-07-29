from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
from xml.sax.saxutils import escape


try:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
except NameError:
    PROJECT_ROOT = Path.cwd().resolve()
    if PROJECT_ROOT.name == "code":
        PROJECT_ROOT = PROJECT_ROOT.parent

NOTES_DIR = PROJECT_ROOT / "notes"
INPUT_MD = NOTES_DIR / "meeting2_elo_summary_2025.md"
OUTPUT_DOCX = NOTES_DIR / "meeting2_elo_summary_2025.docx"


def paragraph_xml(text, style=None, bold=False):
    """Create a simple Word paragraph XML block."""
    style_xml = ""
    if style:
        style_xml = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>'

    bold_xml = "<w:b/>" if bold else ""
    safe_text = escape(text)
    return (
        "<w:p>"
        f"{style_xml}"
        "<w:r>"
        f"<w:rPr>{bold_xml}</w:rPr>"
        f"<w:t xml:space=\"preserve\">{safe_text}</w:t>"
        "</w:r>"
        "</w:p>"
    )


def markdown_to_word_paragraphs(markdown_text):
    """Convert the small meeting markdown file into Word paragraph XML."""
    paragraphs = []

    for raw_line in markdown_text.splitlines():
        line = raw_line.rstrip()

        if not line.strip():
            paragraphs.append(paragraph_xml(""))
            continue

        if line.startswith("# "):
            paragraphs.append(paragraph_xml(line[2:].strip(), style="Heading1", bold=True))
        elif line.startswith("## "):
            paragraphs.append(paragraph_xml(line[3:].strip(), style="Heading2", bold=True))
        elif line.startswith("* "):
            paragraphs.append(paragraph_xml("\u2022 " + line[2:].strip(), style="ListParagraph"))
        elif is_numbered_line(line):
            paragraphs.append(paragraph_xml(line.strip(), style="ListParagraph"))
        else:
            paragraphs.append(paragraph_xml(line.strip()))

    return "\n".join(paragraphs)


def is_numbered_line(line):
    """Return True for lines such as '1. Question text'."""
    stripped = line.strip()
    dot_index = stripped.find(".")
    if dot_index <= 0:
        return False
    return stripped[:dot_index].isdigit() and stripped[dot_index + 1 :].startswith(" ")


def build_document_xml(body_xml):
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    {body_xml}
    <w:sectPr>
      <w:pgSz w:w="12240" w:h="15840"/>
      <w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440"/>
    </w:sectPr>
  </w:body>
</w:document>
"""


def build_styles_xml():
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal">
    <w:name w:val="Normal"/>
    <w:qFormat/>
    <w:rPr><w:sz w:val="22"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading1">
    <w:name w:val="heading 1"/>
    <w:basedOn w:val="Normal"/>
    <w:next w:val="Normal"/>
    <w:qFormat/>
    <w:rPr><w:b/><w:sz w:val="32"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading2">
    <w:name w:val="heading 2"/>
    <w:basedOn w:val="Normal"/>
    <w:next w:val="Normal"/>
    <w:qFormat/>
    <w:rPr><w:b/><w:sz w:val="26"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="ListParagraph">
    <w:name w:val="List Paragraph"/>
    <w:basedOn w:val="Normal"/>
    <w:qFormat/>
    <w:pPr><w:ind w:left="720"/></w:pPr>
  </w:style>
</w:styles>
"""


def create_docx(markdown_path, output_path):
    if not markdown_path.exists():
        raise FileNotFoundError(f"Markdown file not found: {markdown_path}")

    markdown_text = markdown_path.read_text(encoding="utf-8")
    body_xml = markdown_to_word_paragraphs(markdown_text)
    document_xml = build_document_xml(body_xml)

    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>
"""

    package_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>
"""

    document_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>
"""

    core_props = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/">
  <dc:title>2025 Simple Elo Prototype Summary</dc:title>
  <dc:creator>Codex</dc:creator>
</cp:coreProperties>
"""

    app_props = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties">
  <Application>Codex</Application>
</Properties>
"""

    with ZipFile(output_path, "w", ZIP_DEFLATED) as docx:
        docx.writestr("[Content_Types].xml", content_types)
        docx.writestr("_rels/.rels", package_rels)
        docx.writestr("word/document.xml", document_xml)
        docx.writestr("word/styles.xml", build_styles_xml())
        docx.writestr("word/_rels/document.xml.rels", document_rels)
        docx.writestr("docProps/core.xml", core_props)
        docx.writestr("docProps/app.xml", app_props)


def main():
    create_docx(INPUT_MD, OUTPUT_DOCX)
    print(f"Created Word document: {OUTPUT_DOCX}")


if __name__ == "__main__":
    main()
