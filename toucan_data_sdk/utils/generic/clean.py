import re
from unicodedata import normalize


def _slugify(text, delim='-'):
    """ Generate an ASCII-only slug """
    _punctuation_re = re.compile(r'[\t !"#$%&\'()*\-/<=>?@\[\\\]^_`{|},.\n]+')

    result = []
    for word in _punctuation_re.split(text.lower()):
        word = normalize('NFKD', word) \
            .encode('ascii', 'ignore') \
            .decode('utf-8')

        if word:
            result.append(word)

    return delim.join(result)


def get_category_cols(df, threshold):
    obj_df = df.select_dtypes(include=['object'])
    return [col for col in obj_df.columns if len(obj_df[col].unique()) < threshold]


def get_int_cols(df):
    float_df = df.select_dtypes(include=['floating'])
    return [col for col in float_df.columns if all(x.is_integer() for x in float_df[col])]


def clean_dataframe(df, slugify=True, threshold=50, rename_cols=None):
    """
    This method is used to:
    - slugify the column names (if slugify is set to True)
    - convert columns to 'category' (if len(unique) < threshold) or 'int'
    - clean the dataframe and rename if necessary
    """
    if slugify:
        df = df.rename(columns=_slugify)

    df = df.dropna(axis=1, how='all')
    for column in get_category_cols(df, threshold=threshold):
        df[column] = df[column].astype('category')
    for column in get_int_cols(df):
        df[column] = df[column].astype(int)

    if rename_cols is not None:
        df = df.rename(columns=rename_cols)

    return df
