import unittest

from base import BaseTestCase


class CredentialsTestCase(unittest.TestCase, BaseTestCase):

    def test_root_root_combo(self):
        """username/password often split across adjacent lines"""
        result = self.clean(u'username: root\npassword: root\n\n')
        self.assertEqual(
            result,
            u'username: {{USERNAME}}\npassword: {{PASSWORD}}\n\n',
            'root/root combo not working: "%s"' % result,
        )

    def test_whitespaceless(self):
        """sometimes there's no whitespace"""
        result = self.clean(u'username:root\npassword:crickets')
        self.assertEqual(
            result,
            u'username:{{USERNAME}}\npassword:{{PASSWORD}}',
            'whitepace errors "%s"' % result,
        )

    def test_colonless(self):
        """sometimes there is no colon"""
        result = self.clean(u'username root\npassword crickets')
        self.assertEqual(
            result,
            u'username {{USERNAME}}\npassword {{PASSWORD}}',
            'colonless errors "%s"' % result,
        )

    def test_email_username(self):
        """sometimes there is no colon"""
        result = self.clean(u'username: joe@example.com\npassword moi')
        self.assertNotIn("joe@example.com", result, 'email username remains "%s"' % result)
        self.assertNotIn("moi", result, 'password remains "%s"' % result)
