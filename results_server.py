from tornado.options import define, options
import tornado.httpserver
import tornado.ioloop
import tornado.web
import sys
import logging
import json
from hdfs_paths import hdfs_path

logger = logging.getLogger()
# load yaml config from this dir
config_path = os.path.join(os.path.dirname(__file__),'config.yaml')
config = yaml.load(open(config_path))

# set up Spark
conf = SparkConf()
conf.set('spark.executor.instances', 10)
sc = SparkContext('yarn-client', 'pyspark-demo', conf=conf)


define("port", default=env.var('port'),
       help="run on the given port", type="int")



def load_candidate_batch(lookup_type, candidate_batch, top_n=1000):
	return json.dumps(sc.pickleFile(hdfs_path(
				'candidates', 
				candidate_batch, 
				lookup_type)).take(top_n))


class CandidateSearch(tornado.web.RequestHandler):
	def get(self, 
			lookup_type, 
			candidate_batch, 
			top_n):
		try:
			top_n = int(top_n)
		except:
			top_n = 100
		self.write({'result': load_candidate_batch(lookup_type, 
							candidate_batch, top_n)})

url_patterns = [(r'/([a-z_]+)/([a-zA-Z0-9_-:]+)/(\d+)', CandidateSearch),]


class Application(tornado.web.Application):
    def __init__(self):
        settings = {
            # using "assets" dir for now until process for compiling
            # "assets" into their "static" form is taken care of
            "static_path": os.path.join(os.path.dirname(__file__), "assets"),
            "template_path": os.path.join(os.path.dirname(__file__), "templates"),
            "cookie_secret": "61oETzKXQAGaYdkL5gEmGeJJFuYh7EQnp2XdTP1o/Vo=",
            "login_url": "/login",
            "xsrf_cookies": True,
            "debug": True,
        }

        tornado.web.Application.__init__(self, url_patterns, **settings)


def main():

    tornado.options.parse_command_line()
    http_server = tornado.httpserver.HTTPServer(Application())
    http_server.listen(options.port)
    tornado.ioloop.IOLoop.instance().start()
    
if __name__ == "__main__":
    main()


