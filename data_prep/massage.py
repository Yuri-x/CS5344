import io
import typing
from data_prep.config import pgsql_conf
from koala.connector import PGSQLConnector
import math
import random
random.seed(42)
VAL_SPLIT = 0.1

TWEET_COLS = ['tweetid', 'userid', 'user_display_name', 'user_screen_name', 'user_reported_location', 'user_profile_description', 'user_profile_url', 'follower_count', 'following_count', 'account_creation_date', 'account_language', 'tweet_language', 'tweet_text', 'tweet_time', 'tweet_client_name', 'in_reply_to_userid', 'in_reply_to_tweetid', 'quoted_tweet_tweetid', 'is_retweet', 'retweet_userid', 'retweet_tweetid', 'latitude', 'longitude', 'quote_count', 'reply_count', 'like_count', 'retweet_count', 'hashtags', 'urls', 'user_mentions']
USER_COLS = ['userid', 'user_display_name', 'user_screen_name', 'user_reported_location', 'user_profile_description', 'user_profile_url', 'follower_count', 'following_count', 'account_creation_date', 'account_language']
countries = ['bangladesh', 'catalonia', 'china', 'ecuador', 'iran', 'russia', 'saudi_arabia', 'spain', 'uae', 'venezuela']


def rename_table():
    conn = PGSQLConnector(pgsql_conf)
    conn.execute(f"""
    ALTER TABLE "bangladesh_linked_tweets_csv_hashed" RENAME TO "bangladesh_201901_tweets_csv_hashed";
    ALTER TABLE "venezuela_linked_tweets_csv_hashed" RENAME TO "venezuela_201901_2_tweets_csv_hashed";
    """)
    for tablename in conn.list_tables('%082019%'):
        conn.execute(f"""
        ALTER TABLE "{tablename}" RENAME TO "{tablename.replace('082019', '201908')}";
        """)
    for tablename in conn.list_tables('%112019%'):
        conn.execute(f"""
        ALTER TABLE "{tablename}" RENAME TO "{tablename.replace('112019', '201911')}";
        """)
    conn.commit()


def combine_table(target: str, col_list: typing.List[str], target_table: str):
    conn = PGSQLConnector(pgsql_conf)
    first = True
    sio = io.StringIO()
    for country in countries:
        for tablename in conn.list_tables(f"{country}_%{target}%"):
            release = tablename[len(country) + 1:tablename.find('_', len(country) + 1)]
            if not first:
                sio.write("UNION ALL")
            sio.write(f"""
            SELECT '{country}' AS "sponsoring_country", '{release}' AS "release", "{'", "'.join(col_list)}"
            """)
            if first:
                sio.write(f"""
                INTO "{target_table}"
                """)
                first = False
            sio.write(f"""
            FROM "{tablename}"
            """)
            if target == 'tweets':
                sio.write(f"""
                WHERE "tweet_language" = 'en'
                """)
    conn.drop_table(target_table)
    conn.execute(sio.getvalue())
    conn.commit()


def split_data(conn: PGSQLConnector, country: str, dataset_table: str, target_table: str):
    data, _, _, _ = conn.execute(f"""
    SELECT userid, COUNT(tweetid) AS no_of_tweets
    FROM "{dataset_table}"
    WHERE sponsoring_country = '{country}'
    GROUP BY userid
    """, result=True, result_batch=False)
    total_tweets = sum(x[1] for x in data)
    split = int(math.floor(VAL_SPLIT * total_tweets))
    random.shuffle(data)
    current_sum = 0
    for user_id, count in data:
        current_sum += count
        conn.insert(target_table, (country, user_id))
        if current_sum >= split:
            return


def clean_data(country: typing.List[str]):
    conn = PGSQLConnector(pgsql_conf)
    dataset_table = 'dataset'
    conn.drop_table(dataset_table)
    conn.execute(f"""
    SELECT sponsoring_country,
        tweetid,
        userid,
        tweet_text,
        user_profile_url,
        in_reply_to_userid,
        in_reply_to_tweetid,
        quoted_tweet_tweetid,
        retweet_userid,
        retweet_tweetid,
        CASE WHEN hashtags = '[]' THEN NULL ELSE
        RTRIM(LTRIM(hashtags, '['), ']') END AS hashtags,
        CASE WHEN urls = '[]' THEN NULL ELSE
        RTRIM(LTRIM(urls, '['), ']') END AS urls,
        CASE WHEN user_mentions = '[]' THEN NULL ELSE
        RTRIM(LTRIM(user_mentions, '['), ']') END AS user_mentions
    INTO "{dataset_table}"
    FROM "combined_tweets"
    WHERE sponsoring_country IN ('{"', '".join(country)}')
    AND tweet_text IS NOT NULL
    """)
    split_table = 'split_table'
    conn.drop_table(split_table)
    conn.execute(f"""
    SELECT sponsoring_country, userid
    INTO "{split_table}"
    FROM "{dataset_table}"
    LIMIT 0;
    """)
    for c in country:
        split_data(conn, c, dataset_table, split_table)
    split_set = 'dataset_split'
    conn.drop_table(split_set)
    conn.execute(f"""
    SELECT a.*, CAST(CASE WHEN b.userid IS NULL THEN 0 ELSE 1 END AS BOOLEAN) AS "is_validation"
    INTO "{split_set}"
    FROM "{dataset_table}" a
    LEFT JOIN "{split_table}" b
    ON a.sponsoring_country = b.sponsoring_country AND a.userid = b.userid
    """)
    target_table = f"dataset_users"
    conn.drop_table(target_table)
    conn.execute(f"""
    SELECT sponsoring_country,
        tweetid,
        userid,
        user_profile_url,
        in_reply_to_userid,
        in_reply_to_tweetid,
        quoted_tweet_tweetid,
        retweet_userid,
        retweet_tweetid,
        hashtags,
        urls,
        user_mentions,
        is_validation
    INTO "{target_table}"
    FROM "{split_set}"
    """)
    target_table = f"dataset_tweets"
    conn.drop_table(target_table)
    conn.execute(f"""
    SELECT sponsoring_country,
        tweetid,
        userid,
        tweet_text,
        is_validation
    INTO "{target_table}"
    FROM "{split_set}"
    """)
    conn.commit()


def string_to_array(string: str):
    return eval(string)


if __name__ == '__main__':
    rename_table()
    for target, cols in [('tweets', TWEET_COLS), ('users', USER_COLS)]:
        combine_table(target, cols, f"combined_{target}")
    clean_data(['china', 'iran', 'russia', 'venezuela'])

