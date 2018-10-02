import psycopg2  # http://initd.org/psycopg/docs/usage.html
import config


class Database:
    conn: psycopg2.connect()

    def connect(self):
        # url = postgres://{user}:{password}@{hostname}:{port}/{database-name}
        url = 'postgres://' + config.get_value('Heroku', 'User') + ':' +\
            config.get_value('Heroku', 'Password') + '@' + config.get_value('Heroku', 'Host') +\
            config.get_value('Heroku', 'Port') + '/' + config.get_value('Heroku', 'Database')
        print(url)

        self.conn = psycopg2.connect(url, sslmode='require')

    def close(self):
        self.conn.close()
