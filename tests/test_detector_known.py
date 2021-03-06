import unittest

import scrubadub

class PredefinedTestCase(unittest.TestCase):

    def test_simple(self):
        """test a simple matching"""

        test_str = 'this is a test string'
        detector = scrubadub.detectors.KnownFilthDetector([
            {'match': 'test', 'filth_type': 'test'},
        ])

        matches = list(detector.iter_filth(test_str))
        self.assertEquals(matches[0].beg, 10)
        self.assertEquals(matches[0].end, 14)

    def test_empty(self):
        """test a simple matching"""

        test_str = 'this is a test string'

        detector = scrubadub.detectors.KnownFilthDetector()
        matches = list(detector.iter_filth(test_str))
        self.assertEquals(len(matches), 0)

        detector = scrubadub.detectors.KnownFilthDetector([])
        matches = list(detector.iter_filth(test_str))
        self.assertEquals(len(matches), 0)

    def test_start_end(self):
        """text matches with a start and end"""

        test_str = 'hello this is a test string'
        detector = scrubadub.detectors.KnownFilthDetector([
            {'match': 'this is', 'match_end': 'test', 'filth_type': 'test'},
        ])

        matches = list(detector.iter_filth(test_str))
        self.assertEquals(matches[0].beg, 6)
        self.assertEquals(matches[0].end, 20)

    def test_start_no_end(self):
        """text matches with a start and an invalid end"""

        test_str = 'hello this is a test string'
        detector = scrubadub.detectors.KnownFilthDetector([
            {'match': 'this is', 'match_end': 'impossible to find', 'filth_type': 'test'},
        ])

        matches = list(detector.iter_filth(test_str))
        self.assertEquals(len(matches), 0)

    def test_error(self):
        """text exceptions thrown by predefined"""

        with self.assertRaises(KeyError):
            detector = scrubadub.detectors.KnownFilthDetector([
                {'non_existiant': 'this is'},
            ])

        with self.assertRaises(KeyError):
            detector = scrubadub.detectors.KnownFilthDetector([
                {'match': 'the match', 'filth_type': 'email', 'non_existiant': 'this is'},
            ])

    def test_filth_string(self):
        filth = scrubadub.filth.KnownFilth(beg=0, end=5)
        self.assertEqual(str(filth), "<KnownFilth text='' beg=0 end=5>")

        filth = scrubadub.filth.KnownFilth(beg=0, end=5, text='hello')
        self.assertEqual(str(filth), "<KnownFilth text='hello' beg=0 end=5>")

        filth = scrubadub.filth.KnownFilth(beg=0, end=5, text='hello', document_name='hello.txt')
        self.assertEqual(str(filth), "<KnownFilth text='hello' document_name='hello.txt' beg=0 end=5>")

        filth = scrubadub.filth.KnownFilth(beg=0, end=5, text='hello', comparison_type='greeting')
        self.assertEqual(str(filth), "<KnownFilth text='hello' beg=0 end=5 comparison_type='greeting'>")

        filth = scrubadub.filth.KnownFilth(beg=0, end=5, text='hello', document_name='hello.txt',
                                           comparison_type='greeting')
        self.assertEqual(
            str(filth),
            "<KnownFilth text='hello' document_name='hello.txt' beg=0 end=5 comparison_type='greeting'>"
        )
