from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'

if __name__ == '__main__':
    app.run(debug=True)



# @app.route('/process-image', methods=['POST'])
# def process_image():
#     # Get the image data from the request
#     image_data = request.data
#
#     # Run your computer vision code on the image data
#     # (Replace this with your actual code)
#     result = run_my_cv_code(image_data)
#
#     # Return the result as JSON
#     return jsonify(result)
