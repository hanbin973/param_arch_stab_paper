.SECONDARY :

paper.pdf : preamble.sty references.bib plr_refs.bib img/diagram2.pdf review-responses.tex

%.pdf : %.tex %.bbl
	while ( pdflatex $<;  grep -q "[rR]erun \(to get\|LaTeX\)" $*.log ) do true ; done
	touch $*.bbl
	touch $@

%.bcf %.aux : %.tex
	-pdflatex $<

%.bbl : %.bcf references.bib plr_refs.bib
	biber $<

# %.bbl : %.aux references.bib plr_refs.bib
# 	bibtex $<

%.png : %.pdf
	convert -density 300 $< -flatten $@

%.pdf : %.svg
	inkscape $< --export-area-drawing --export-filename=$@

%.pdf : %.eps
	# inkscape $< --export-filename=$@
	epspdf $<

%.pdf : %.ink.svg
	inkscape $< --export-filename=$@

paper-diff%.tex : paper.tex
	latexdiff-git --force --flatten -r $* $<

diff-to-submitted-version.pdf : paper-diff316ccbaed4e1e0f.pdf
	cp $< $@

.SECONDARY : 
