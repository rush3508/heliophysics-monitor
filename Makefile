# Heliophysics Monitor — Makefile
#
# Targets:
#   make all          Full pipeline (ingest → dashboard)
#   make ingest       Fetch DONKI events only
#   make features     Compute aggregates only
#   make corpus       One-off: scrape + embed corpus
#   make brief        Generate analyst brief only
#   make dashboard    Rebuild static dashboard
#   make clean        Remove all generated files

VENV := .venv/bin/python
BACKFILL := 180

.PHONY: all ingest features corpus brief dashboard clean

all:
	$(VENV) main.py --backfill $(BACKFILL)

ingest:
	$(VENV) main.py --ingest-only --backfill $(BACKFILL)

features:
	$(VENV) main.py --features-only

corpus:
	$(VENV) main.py --build-corpus

brief:
	$(VENV) main.py --brief-only

dashboard:
	$(VENV) main.py --dashboard-only

clean:
	rm -rf data/raw/*.json
	rm -rf data/processed/*.json
	rm -rf data/corpus/*.txt
	rm -rf docs/corpus/chunks.json
	rm -rf docs/corpus/chroma/
	rm -rf dashboard/index.html
	rm -rf dashboard/data/*.json
	rm -rf reports/brief.md
	rm -rf reports/report_*.md
	rm -rf reports/latest.md
	rm -rf reports/pipeline_warnings.log
	@echo "Cleaned all generated files."
