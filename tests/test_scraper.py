from src.ingest.scraper import parse_links

SAMPLE = """
<html><body>
<a href="https://f001.backblazeb2.com/file/Backblaze-Hard-Drive-Data/data_Q1_2023.zip">Q1 2023</a>
<a href="https://f001.backblazeb2.com/file/Backblaze-Hard-Drive-Data/data_Q2_2023.zip">Q2 2023</a>
<a href="/file/Backblaze-Hard-Drive-Data/data_2014.zip">2014</a>
<a href="https://f001.backblazeb2.com/file/Backblaze-Hard-Drive-Data/data_Q1_2023.zip">dup</a>
<a href="/blog/some-post">not a zip</a>
</body></html>
"""


def test_parse_links_finds_zips_dedupes_and_sorts():
    entries = parse_links(SAMPLE)
    assert [e["file_name"] for e in entries] == [
        "data_2014.zip", "data_Q1_2023.zip", "data_Q2_2023.zip"
    ]
    assert entries[0]["quarter"] is None and entries[0]["year"] == 2014
    assert entries[1]["quarter"] == 1 and entries[1]["year"] == 2023
    assert all(e["status"] == "pending" for e in entries)


def test_relative_links_become_absolute():
    entries = parse_links(SAMPLE)
    assert entries[0]["url"].startswith("https://www.backblaze.com/")
