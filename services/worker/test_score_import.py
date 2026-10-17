import unittest

import numpy as np

from score_import import detect_staffs


class ScoreImportTest(unittest.TestCase):
    def test_detects_two_six_line_tab_staffs(self) -> None:
        page = np.full((420, 900), 255, dtype=np.uint8)
        for top in (70, 250):
            for string_index in range(6):
                y = top + string_index * 18
                page[y : y + 2, 40:860] = 20
            page[top - 3 : top + 94, 40:43] = 20
            page[top - 3 : top + 94, 857:860] = 20

        staffs = detect_staffs(page, page=0)

        self.assertEqual(len(staffs), 2)
        self.assertEqual(staffs[0].lines, (70, 88, 106, 124, 142, 160))
        self.assertEqual(staffs[1].lines, (250, 268, 286, 304, 322, 340))
        self.assertGreater(staffs[0].right - staffs[0].left, 800)


if __name__ == '__main__':
    unittest.main()
