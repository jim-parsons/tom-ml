from flask import Flask
from flask_restplus import Api, Resource, fields

app = Flask(__name__)
api = Api(app)

a_language = api.model('language', {
    'language': fields.String('TheLanguage')
})

languages = list()
python = {'language': 'python'}
languages.append(python)


@api.route('/language')
class language(Resource):
    def get(self):
        return languages

    @api.expect(a_language)
    def post(self):
        languages.append(api.payload)
        return {'result': 'language added'}, 201


if __name__ == '__main__':
    app.run(debug=True)
