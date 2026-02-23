#!/usr/bin/env python3
"""
KardiaMobile 1-lead file format converter from Kardia ECG PDF to EDF.

This script reads the vector path data embedded in the PDF, converts
the path coordinates to voltage values using the known calibration
(10mm/mV, 25mm/s), and writes the result as a standard EDF file.

Metadata extracted from PDF:
  - Patient: Joel Henderson
  - DOB: 5/4/70, Sex: Male, Age: 55
  - Recorded: 2026-02-13 at 22:42:00
  - Heart Rate: 76 BPM
  - Duration: 30s
  - 1 lead: Lead I
  - Sampling rate: 300 Hz
  - Kardia Determination: Normal Sinus Rhythm
"""

from datetime import datetime

import numpy as np
import pyedflib
import pymupdf


def extract_baselines(page):
    """Extract the baseline y-coordinates for each row from horizontal grid lines.

    The 1-lead PDF displays the single lead across multiple rows on one page.
    Each row has a horizontal baseline at its center.
    """
    paths = page.get_drawings()
    for path in paths:
        items = path.get("items", [])
        color = path.get("color")
        width = path.get("width", 0)
        # The baseline path has horizontal lines spanning the page, width ~0.4
        if (
            color == (0.0, 0.0, 0.0)
            and width is not None
            and 0.35 < width < 0.45
            and len(items) >= 4
        ):
            y_values = []
            for item in items:
                if item[0] == "l":
                    p1, p2 = item[1], item[2]
                    if abs(p1.y - p2.y) < 0.01 and abs(p2.x - p1.x) > 500:
                        y_values.append(p1.y)
            # Only keep baselines that are within the visible page area (y < 760)
            visible = [y for y in y_values if y < 760]
            if len(visible) >= 4:
                return visible[:4]  # 4 baselines for 4 rows
    return None


def extract_ecg_waveform_rows(page, baselines):
    """Extract ECG waveform points grouped by row.

    For a 1-lead PDF, the single lead is displayed across multiple rows,
    each representing a consecutive time segment.

    Returns dict: row_index -> list of (x, y) points sorted by x.
    """
    paths = page.get_drawings()
    rows = {i: [] for i in range(len(baselines))}

    for path in paths:
        color = path.get("color")
        width = path.get("width", 0)
        items = path.get("items", [])

        # ECG waveform paths: black, width ~0.4, many line segments
        if (
            color != (0.0, 0.0, 0.0)
            or width is None
            or not (0.35 < width < 0.45)
            or len(items) < 40
        ):
            continue

        # Skip single horizontal lines (baselines, separators)
        if len(items) == 1:
            continue

        # Extract points from line segments
        points = []
        for item in items:
            if item[0] == "l":
                p1, p2 = item[1], item[2]
                if (
                    not points
                    or abs(points[-1][0] - p1.x) > 0.001
                    or abs(points[-1][1] - p1.y) > 0.001
                ):
                    points.append((p1.x, p1.y))
                points.append((p2.x, p2.y))

        if not points:
            continue

        # Determine which row by y-center proximity to baselines
        y_center = np.mean([p[1] for p in points])
        min_dist = float("inf")
        best_row = -1
        for ri, bl in enumerate(baselines):
            dist = abs(y_center - bl)
            if dist < min_dist:
                min_dist = dist
                best_row = ri

        if best_row >= 0 and min_dist < 80:
            rows[best_row].extend(points)

    # Sort each row's points by x-coordinate
    for ri in rows:
        rows[ri].sort(key=lambda p: p[0])

    return rows


def points_to_voltage(points, baseline_y, cal_pt_per_mv):
    """Convert (x, y) points to voltage values in millivolts.

    In the PDF, y increases downward, so voltage = (baseline - y) / scale.
    """
    voltages = [(baseline_y - y) / cal_pt_per_mv for _, y in points]
    return voltages


def main():
    pdf_path = "/Users/jph/git/joelparkerhenderson/kardiamobile-personal-health-information/kardiamobile-1-lead-ecg.pdf"
    edf_path = "/Users/jph/git/joelparkerhenderson/kardiamobile-personal-health-information/kardiamobile-1-lead-ecg.edf"

    doc = pymupdf.open(pdf_path)

    # Calibration: 1 mV = 28.346 PDF points (10mm at 2.8346 pt/mm)
    CAL_PT_PER_MV = 28.346
    SAMPLE_RATE = 300  # Hz
    LEAD_NAME = "I"

    # Extract baselines from page 2 (index 1)
    page = doc[1]
    baselines = extract_baselines(page)
    if not baselines:
        raise RuntimeError("Could not find baseline grid lines in PDF")

    print(f"Baselines (PDF y-coordinates): {[f'{b:.1f}' for b in baselines]}")

    # Extract waveform rows from page 2
    rows = extract_ecg_waveform_rows(page, baselines)

    doc.close()

    # Concatenate all rows into a single voltage signal (top-to-bottom = chronological)
    all_voltages = []
    for ri in range(len(baselines)):
        points = rows[ri]
        if not points:
            print(f"Row {ri}: no data")
            continue

        # Remove duplicate x-coordinates (boundary points between segments)
        deduped = [points[0]]
        for i in range(1, len(points)):
            if abs(points[i][0] - deduped[-1][0]) > 0.01:
                deduped.append(points[i])

        voltages = points_to_voltage(deduped, baselines[ri], CAL_PT_PER_MV)
        print(
            f"Row {ri}: {len(voltages)} samples, "
            f"x:[{deduped[0][0]:.1f}-{deduped[-1][0]:.1f}], "
            f"range [{min(voltages):.3f}, {max(voltages):.3f}] mV"
        )
        all_voltages.extend(voltages)

    signal = np.array(all_voltages, dtype=np.float64)
    duration_sec = len(signal) / SAMPLE_RATE

    print(f"\nTotal samples: {len(signal)}")
    print(f"Duration: {duration_sec:.2f} seconds")
    print(f"Sampling rate: {SAMPLE_RATE} Hz")
    print(f"Voltage range: [{signal.min():.3f}, {signal.max():.3f}] mV")

    # Write EDF file
    n_channels = 1
    edf_writer = pyedflib.EdfWriter(
        edf_path, n_channels, file_type=pyedflib.FILETYPE_EDFPLUS
    )

    # Patient and recording info
    edf_writer.setPatientName("Joel_Henderson")
    edf_writer.setSex(1)  # 1 = male
    edf_writer.setBirthdate(datetime(1970, 5, 4).date())
    edf_writer.setStartdatetime(datetime(2026, 2, 13, 22, 42, 0))
    edf_writer.setEquipment("KardiaMobile_1L")
    edf_writer.setRecordingAdditional("Normal_Sinus_Rhythm_HR_76_BPM")

    # Configure channel
    edf_writer.setLabel(0, f"EKG {LEAD_NAME}")
    edf_writer.setPhysicalDimension(0, "mV")
    edf_writer.setSamplefrequency(0, SAMPLE_RATE)
    edf_writer.setTransducer(0, "KardiaMobile 1L electrode")
    edf_writer.setPrefilter(0, "Enhanced Filter, 50Hz mains")

    # Set physical min/max from actual data with margin
    phys_min = float(np.min(signal)) - 0.1
    phys_max = float(np.max(signal)) + 0.1
    edf_writer.setPhysicalMinimum(0, phys_min)
    edf_writer.setPhysicalMaximum(0, phys_max)
    edf_writer.setDigitalMinimum(0, -32768)
    edf_writer.setDigitalMaximum(0, 32767)

    # Write data
    edf_writer.writeSamples([signal])
    edf_writer.close()

    print(f"\nEDF file written: {edf_path}")
    print(f"File size: {__import__('os').path.getsize(edf_path):,} bytes")


if __name__ == "__main__":
    main()
