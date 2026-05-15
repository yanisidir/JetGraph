TEX = pdflatex
BIB = bibtex
TARGET = main

.PHONY: all clean

all: $(TARGET).pdf

$(TARGET).pdf: $(TARGET).tex references.bib
	$(TEX) $(TARGET)
	$(BIB) $(TARGET)
	$(TEX) $(TARGET)
	$(TEX) $(TARGET)

clean:
	rm -f $(TARGET).aux $(TARGET).bbl $(TARGET).blg $(TARGET).log
	rm -f $(TARGET).out $(TARGET).toc $(TARGET).fls $(TARGET).fdb_latexmk
