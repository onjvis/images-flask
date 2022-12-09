from flask import Flask, request
from flask_restful import Resource, Api
from flask_cors import CORS

import api_utils
import file_manipulation
import hist
from model.color_model import ColorModel

app = Flask(__name__)
api = Api(app)
CORS(app)


class Image(Resource):
    def post(self):
        # request parsing
        req = request.get_json()
        comparison_method = req['method']
        comparison_color_model = req['colorModel']
        comparison_bin_count = req['binCount']
        filename = req['filename']

        # extra parameters for custom methods
        params = api_utils.parse_args(request.args)

        # save an image
        image_path = file_manipulation.save_image_from_b64(req['filename'], req['image'])

        # for each bin count and each color model compute histogram, normalize it and save it
        for color_model in [ColorModel.RGB, ColorModel.HSV]:
            for bin_count in [16, 32, 64, 128, 256]:
                h = hist.compute_histogram(color_model, bin_count, image_path)
                h_norm = hist.normalize_histogram(h)
                hist.save_histogram(color_model, bin_count, filename, h_norm)

        # load histogram based on the input params
        hist_path = hist.get_histogram_filepath(comparison_color_model, comparison_bin_count, filename)
        histogram = hist.load_histogram(f'{hist_path}{hist.BASE_EXT}')

        # compare input image with the other images stored at the server
        h_comp = hist.compare(histogram, color_model=comparison_color_model, method=comparison_method,
                              bin_count=comparison_bin_count, params=params)

        # sort results by their respective score
        sorted_comp = hist.sort(h_comp)

        # remove the queried image itself
        sorted_comp = dict(sorted_comp)
        sorted_comp.pop(file_manipulation.remove_extension(filename))
        sorted_comp = list(tuple(sorted_comp.items()))

        # take first 10 results
        if len(sorted_comp) > 10:
            sorted_comp = [sorted_comp[i] for i in range(10)]
        results = []
        for result in sorted_comp:
            image = file_manipulation.find_image(result[0])
            content = file_manipulation.convert_to_b64(image)
            results.append({"image": str(content), "score": str(result[1])})

        # finally return a list of similar images
        return results


api.add_resource(Image, '/')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
