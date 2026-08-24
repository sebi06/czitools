"""Throwaway probe: enumerate stored pyramid zooms across CZI test files."""

from pylibCZIrw import czi as pyczi

files = [
    "data/CellDivision_T3_Z5_CH2_X240_Y170.czi",
    "data/Tumor_HE_RGB.czi",
    "data/WellD6_S1.czi",
    r"F:\Testdata_Zeiss\CZI_Testfiles\DTScan_ID4.czi",
    r"F:\Testdata_Zeiss\CD7\Mouse Kidney_40x0.95_3CD_JK_comp.czi",
]


def summarise(fp: str) -> None:
    print(f"\n=== {fp} ===")
    zooms: dict[float, int] = {}
    sizes: dict[float, tuple[int, int, int, int]] = {}

    def _cb(idx: int, info) -> bool:
        try:
            z = float(info.get_zoom())
        except Exception as exc:
            print(f"  get_zoom failed at idx {idx}: {exc}")
            return False
        zooms[z] = zooms.get(z, 0) + 1
        if z not in sizes:
            lr = info.logicalRect
            ps = info.physicalSize
            sizes[z] = (lr.w, lr.h, ps.w, ps.h)
        return sum(zooms.values()) < 5000

    try:
        with pyczi.open_czi(fp) as doc:
            doc.enumerate_subblocks(_cb)
    except Exception as e:
        print("open failed:", e)
        return

    for z, n in sorted(zooms.items(), reverse=True):
        lw, lh, pw, ph = sizes[z]
        print(f"  zoom={z:.6f}  count={n:5d}  logical={lw}x{lh}  physical={pw}x{ph}")


for f in files:
    summarise(f)
