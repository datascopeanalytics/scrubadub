import re
import nltk

from sklearn.feature_extraction import DictVectorizer as SklDictVectorizer
from sklearn.preprocessing import LabelEncoder as SklLabelEncoder
from sklearn.linear_model import LogisticRegression as SklLogisticRegression
from sklearn.base import BaseEstimator

from typing import Optional

from . import register_detector
from .base import Detector
from ..filth.address import AddressFilth


class EstimatorToJsonMixin(object):
    """This is a utility clas to help serialise sklearn objects to JSON.

    To use, before training make a derrived class  using this mixin, update the attributes_to_save and train as normal.
    Finally, after successfully training your model call model.to_json('output.json').
    The created file can be included in scrubadub for identifying PII.
    """
    # attributes_to_save = ['C', 'max_iter', 'solver', 'X_train', 'Y_train']

    def save_json(self, file_path: str):
        dict_ = {}
        for item_name, item in self.__dict__.items():
            if item_name == 'dtype':
                dict_[item_name] = item.__name__
            elif isinstance(item, np.ndarray):
                dict_[item_name] = item.tolist()
            else:
                dict_[item_name] = item

        json_txt = json.dumps(dict_, indent=4)
        with open(file_path, 'w') as file:
            file.write(json_txt)

    @classmethod
    def load_json(cls, filepath):
        with open(filepath, 'r') as file:
            dict_ = json.load(file)

        new_obj = cls()
        for item_name, new_item in dict_.items():
            existing_item = new_obj.__getattribute__(item_name)
            if item_name == 'dtype':
                new_obj.__setattr__(item_name, np.__getattribute__(new_item))
            elif isinstance(existing_item, np.ndarray):
                new_obj.__setattr__(item_name, np.array(new_item))
            else:
                new_obj.__setattr__(item_name, new_item)

        return new_obj


class DictVectorizer(EstimatorToJsonMixin, SklDictVectorizer):
    # attributes_to_save = ['separator', 'sparse', 'sort', 'dtype', 'feature_names_', 'vocabulary_']
    pass


class LabelEncoder(EstimatorToJsonMixin, SklLabelEncoder):
    # attributes_to_save = ['classes_']
    pass


class LogisticRegression(EstimatorToJsonMixin, SklLogisticRegression):
    # attributes_to_save = ['penalty', 'dual', 'tol', 'C', 'fit_intercept', 'intercept_scaling', 'class_weight',
    #                       'random_state', 'solver', 'max_iter', 'multi_class', 'verbose', 'warm_start', 'n_jobs',
    #                       'l1_ratio', 'classes_', 'coef_', 'intercept_', '', '', '', '', '', '', '', '', '', '', '', '', ]
    pass


class SklearnDetector(Detector):
    """...
    """
    name = 'sklearn'
    punct_languages = {
        'cs': 'czech',
        'da': 'danish',
        'nl': 'dutch',
        'en': 'english',
        'et': 'estonian',
        'fi': 'finnish',
        'fr': 'french',
        'de': 'german',
        'el': 'greek',
        'it': 'italian',
        'no': 'norwegian',
        'pl': 'polish',
        'pt': 'portuguese',
        'ru': 'russian',
        'sl': 'slovene',
        'es': 'spanish',
        'sv': 'swedish',
        'tr': 'turkish',
    }
    label_filth_map = {
        'ADD': AddressFilth,
    }

    def __init__(self, **kwargs):
        super(SklearnDetector, self).__init__(**kwargs)
        self.tokeniser = nltk.tokenize.destructive.NLTKWordTokenizer()
        self.dict_vectorizer_json_path = ''
        self.model_json_path = ''
        self.label_encoder_json_path = ''
        self.dict_vectorizer = None  # type: Optional[BaseEstimator]
        self.model = None  # type: Optional[BaseEstimator]
        self.label_encoder = None  # type: Optional[BaseEstimator]

        language, region = cls.locale_split(self.locale)
        try:
            self.punct_language = self.punct_languages[language]
        except KeyError:
            raise ValueError('The locale is not supported by punct and so this detector cannot be used')

    def word_tokenize(text: str) -> List[str]:
        sentences = nltk.tokenize.sent_tokenize(text, language=self.punct_language)
        return [
            token for sent in sentences for token in self.tokeniser.tokenize(sent)
        ]

    @staticmethod
    def create_features_single_token(token: str, prefix: str = '') -> Dict:
        token = token.strip(' \t\r\v\f')
        features = {
            prefix + 'capitalised': token.istitle(),
            prefix + 'lower': token.islower(),
            prefix + 'upper': token.isupper(),
            prefix + 'numeric': token.isdigit(),
            prefix + 'alphanumeric': any(c.isdigit() for c in token) and any(c.isalpha() for c in token),
            prefix + 'length_long': len(token) >= 12,
            prefix + 'length_short': len(token) <= 5,
        }
        return features

    @staticmethod
    def create_features_with_context(features: List[Dict], prev_features: List[Dict], next_features: List[Dict]) -> Dict:
        full_feature_set = copy.copy(features)

        len_prev = len(prev_features)
        for i, feature_items in enumerate(prev_features):
            prefix = "prev_{:#02d}_".format(len_prev - i)
            full_feature_set.update({prefix + k: v for k, v in feature_items.items()})

        for i, feature_items in enumerate(next_features):
            prefix = "next_{:#02d}_".format(i + 1)
            full_feature_set.update({prefix + k: v for k, v in feature_items.items()})

        return full_feature_set

    def create_features(self, tagged_documents, n_prev_tokens=3, n_next_tokens=5):
        feature_list = [(t[0], self.create_features_single_token(t[1])) for t in tagged_documents]
        all_features = []
        for i, (doc_id, tok, tag) in enumerate(tagged_documents):
            prev_features = [
                t[1]
                for t in feature_list[i - n_prev_tokens if i - n_prev_tokens >= 0 else 0:i]
                if t[0] == doc_id
            ]
            next_features = [
                t[1]
                for t in feature_list[i + 1:i + 1 + n_next_tokens]
                if t[0] == doc_id
            ]

            all_features.append(
                self.create_features_with_context(feature_list[i][1], prev_features, next_features)
            )
        return all_features

    def iter_filth(self, text, document_name: Optional[str] = None):
        text_tokens = [(0, x, 'U') for x in word_tokenize(text)]
        text_features = self.create_features(text_tokens, n_prev_tokens=3, n_next_tokens=5)

        if self.dict_vectorizer is None:
            self.dict_vectorizer = DictVectorizer.load_json(self.dict_vectorizer_json_path)

        if self.model is None:
            self.model = LogisticRegression.load_json(self.model_json_path)

        if self.label_encoder is None:
            self.label_encoder = LabelEncoder.load_json(self.label_encoder_json_path)

        text_data = self.dict_vectorizer.transform(text_features)
        text_prediction = self.model.predict(text_data)
        text_labels = self.label_encoder.inverse_transform(y_predict)

        # TODO: logic to join filth tags
        # TODO: to find filth in text
        for found in found_address_tokens:
            # TODO: logic to map tags to filth classes
            filth_cls
            yield AddressFilth(

            )

    @classmethod
    def supported_locale(cls, locale: str) -> bool:
        """Returns true if this ``Detector`` supports the given locale.

        :param locale: The locale of the documents in the format: 2 letter lower-case language code followed by an
                       underscore and the two letter upper-case country code, eg "en_GB" or "de_CH".
        :type locale: str
        :return: ``True`` if the locale is supported, otherwise ``False``
        :rtype: bool
        """
        language, region = cls.locale_split(locale)
        return language in self.punct_languages.keys()

register_detector(SklearnDetector, autoload=False)
