from pathlib import Path
import io
import json
import hashlib
import zipfile
import pandas as pd
import pytest
from src.reporting.data import read_table, filter_biomarkers, report_html, report_bundle
ROOT=Path(__file__).resolve().parents[1]

def test_real_sample_alignment():
    pca=read_table(ROOT,'pca');clusters=read_table(ROOT,'clusters')
    assert len(pca)==72
    assert pca.set_index('sample_id').cluster.sort_index().equals(clusters.set_index('sample_id').cluster.sort_index())
    assert pca.cluster.value_counts().to_dict()=={1:35,2:23,0:14}

def test_missing_optional_results(tmp_path):
    assert read_table(tmp_path,'enrichment').empty
    assert read_table(tmp_path,'shap').empty

def test_bad_schema_is_rejected(tmp_path):
    path=tmp_path/'reports/enrichment';path.mkdir(parents=True)
    (path/'pathway_results.csv').write_text('wrong\n1\n')
    with pytest.raises(ValueError,match='missing columns'):read_table(tmp_path,'enrichment')

def test_invalid_probability_is_rejected(tmp_path):
    path=tmp_path/'reports/enrichment';path.mkdir(parents=True)
    (path/'pathway_results.csv').write_text('cluster,pathway,gene_set,fdr,overlap_genes\n0,test,GO,2,A\n')
    with pytest.raises(ValueError,match='between 0 and 1'):read_table(tmp_path,'enrichment')

def test_filters_are_literal_and_match_expected_gene():
    frame=read_table(ROOT,'biomarkers')
    result=filter_biomarkers(frame,[0],'CBFB')
    assert len(result)==1
    assert result.iloc[0].gene_symbol=='CBFB'
    assert filter_biomarkers(frame,[], '').empty
    assert filter_biomarkers(frame,[0,1,2],'[').empty

def test_report_escapes_untrusted_cells():
    report=report_html({'genes':pd.DataFrame({'gene':['<script>alert(1)</script>']})})
    assert '<script>' not in report
    assert '&lt;script&gt;' in report

def test_bundle_contains_verifiable_original_data():
    bundle=report_bundle(ROOT,{'clusters':read_table(ROOT,'clusters')})
    with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
        manifest=json.loads(archive.read('manifest.json'))
        for path,digest in manifest['sha256'].items():
            assert hashlib.sha256(archive.read('results/'+path)).hexdigest()==digest
        assert b'Not available' not in archive.read('results/validation/consensus_k3_clusters.csv')
