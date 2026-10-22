import unittest

from qqmusic_windows import (
    extract_qqmusic_windows_credential,
    parse_qqmusic_windows_uin,
)


class QQMusicWindowsCredentialTest(unittest.TestCase):
    def test_reads_current_account_from_utf8_and_utf16_config(self) -> None:
        config = "[History]\nUin=87654321\n[Account]\nUin=12345678\n"
        self.assertEqual(
            parse_qqmusic_windows_uin(config.encode("utf-8")),
            "12345678",
        )
        self.assertEqual(
            parse_qqmusic_windows_uin(
                b"\xff\xfe" + config.encode("utf-16le"),
            ),
            "12345678",
        )

    def test_extracts_current_account_json_and_form_sessions(self) -> None:
        json_session = (
            b'noise\x00{"authst":"Q_H_L_current-session-token",'
            b'"uin":"12345678","tmeLoginType":"3"}\x00noise'
        )
        self.assertEqual(
            extract_qqmusic_windows_credential(
                json_session,
                "12345678",
            ),
            {
                "musicId": "12345678",
                "musicKey": "Q_H_L_current-session-token",
                "loginType": 3,
            },
        )
        form_session = (
            "qqmusic_key=W_X_current-session-token;"
            "qqmusic_uin=12345678;tmeLoginType=1"
        )
        self.assertEqual(
            extract_qqmusic_windows_credential(
                form_session.encode("utf-16le"),
                "12345678",
            ),
            {
                "musicId": "12345678",
                "musicKey": "W_X_current-session-token",
                "loginType": 1,
            },
        )

    def test_extracts_escaped_json_session(self) -> None:
        escaped = (
            rb'{\"authst\":\"Q_H_L_escaped-session-token\",'
            rb'\"uin\":\"12345678\",\"tmeLoginType\":\"3\"}'
        )
        self.assertEqual(
            extract_qqmusic_windows_credential(escaped, "12345678"),
            {
                "musicId": "12345678",
                "musicKey": "Q_H_L_escaped-session-token",
                "loginType": 3,
            },
        )

    def test_rejects_unbound_or_different_account_tokens(self) -> None:
        self.assertIsNone(
            extract_qqmusic_windows_credential(
                b'{"authst":"Q_H_L_unbound-session-token"}',
                "12345678",
            ),
        )
        self.assertIsNone(
            extract_qqmusic_windows_credential(
                b'{"authst":"Q_H_L_other-session-token","uin":"87654321"}',
                "12345678",
            ),
        )
        self.assertIsNone(
            extract_qqmusic_windows_credential(
                b"randomBase64Token1234567890=",
                "12345678",
            ),
        )


if __name__ == "__main__":
    unittest.main()
