import unittest

import torch

from stringtrace_ml.noise2fret import (
    Noise2FretRuntime,
    _audio_features,
)


class Noise2FretAdapterTest(unittest.TestCase):
    def test_inference_adapter_emits_two_six_string_position_slots(self) -> None:
        runtime = Noise2FretRuntime(torch.device("cpu"))
        runtime.eval()
        audio = torch.zeros((1, 16_000, 1), dtype=torch.float32)

        prediction = runtime.predict(audio, _audio_features(audio))

        self.assertEqual(tuple(prediction.shape), (1, 2, 6))
        self.assertTrue(torch.all(prediction >= 0))
        self.assertTrue(torch.all(prediction < 24))


if __name__ == "__main__":
    unittest.main()
