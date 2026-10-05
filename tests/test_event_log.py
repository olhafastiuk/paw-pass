import csv

from src.runtime.event_log import CSV_FIELDS, EventLogger


def test_creates_csv_with_header(tmp_path):
    EventLogger(log_dir=tmp_path, save_frames=False)
    with (tmp_path / "events.csv").open() as f:
        assert next(csv.reader(f)) == CSV_FIELDS


def test_logs_one_row_per_event(tmp_path):
    logger = EventLogger(log_dir=tmp_path, save_frames=False)
    logger.log("open", "mobilenet", 0.912345)
    logger.log("close", "mobilenet", 0.04)

    with (tmp_path / "events.csv").open() as f:
        rows = list(csv.DictReader(f))

    assert [r["event"] for r in rows] == ["open", "close"]
    assert rows[0]["confidence"] == "0.9123"   # rounded to 4 decimals
    assert rows[0]["frame_file"] == ""         # no frame passed


def test_appends_to_an_existing_log(tmp_path):
    EventLogger(log_dir=tmp_path, save_frames=False).log("open", "yolo", 0.7)
    EventLogger(log_dir=tmp_path, save_frames=False).log("close", "yolo", 0.1)

    with (tmp_path / "events.csv").open() as f:
        assert len(list(csv.DictReader(f))) == 2
