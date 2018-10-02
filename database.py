import psycopg2  # http://initd.org/psycopg/docs/usage.html
import config


class Database:

    def __init__(self):
        url = 'postgres://' + config.get_value('Heroku', 'User') + ':' + \
              config.get_value('Heroku', 'Password') + '@' + config.get_value('Heroku', 'Host') + ':' + \
              config.get_value('Heroku', 'Port') + '/' + config.get_value('Heroku', 'Database')

        self.conn = psycopg2.connect(url, sslmode='require')
        self.cur = self.conn.cursor()

        query = ('SELECT table_schema,table_name'
                 'FROM information_schema.tables '
                 'ORDER BY table_schema,table_name;')
        tables = self.execute(query)
        if tables is None:
            self.create_sts_table()

    # just implement and see how well it works! stop slacking
    def create_sts_table(self):
        query = ('CREATE TABLE sts_sentences('
                 'id serial PRIMARY KEY,'
                 'text1 VARCHAR(50) NOT NULL,'
                 'text2 VARCHAR(50) NOT NULL,'
                 'similarity SMALLINT NOT NULL'  # human annotated similarity
                 ');')
        self.execute(query)

        query = ('CREATE TABLE sts_results('
                 'id serial PRIMARY KEY,'
                 'system VARCHAR(50) NOT NULL,'  # system for now defined as string
                 'similarity'  # similarity estimate as provided by system
                 'updated timestamp default current_timestamp'
                 ');')
        self.execute(query)

    def execute(self, query):
        return self.cur.execute(query)

    def close(self):
        self.cur.close()
        self.conn.close()
