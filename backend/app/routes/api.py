from flask import Blueprint, request, jsonify
from app.services.neural_network import train_network, get_network_metrics

api_bp = Blueprint('api', __name__)

@api_bp.route('/test', methods=['GET'])
def test():
    print('Test endpoint hit')
    return jsonify({'status': 'ok'})

@api_bp.route('/metrics', methods=['GET'])
def get_metrics():
    try:
        return jsonify(get_network_metrics())
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@api_bp.route('/train', methods=['POST'])
def train():
    data = request.json
    return jsonify(train_network(data)) 