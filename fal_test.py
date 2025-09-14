import json
from tools.video_generation import generate_scene_videos_tool

# Test with the EXACT structure from the error log
test_input = {
    'image_generation_result': {
        'scenes': [
            {
                'scene_id': 1,
                'image_url': 'https://replicate.delivery/xezq/C4jlRhe0LfhZhUDCwB1UjBmAwaeKPgu7b5h2eMC7pIlxGiOVB/tmp178ibfns.jpeg'
            },
            {
                'scene_id': 2,
                'image_url': 'https://replicate.delivery/xezq/MWnxMHRL4wIBBlmy209tmIscHJSOENo4Bttn7v1HqvB6UF/tmpphf0qybq.jpeg'
            },
            {
                'scene_id': 3,
                'image_url': 'https://replicate.delivery/xezq/fh1QUaNoK5zkBa8QyzFZqqIUUIpCfMoqZCyZJKZHjfiZDRnqA/tmpzqf7bh85.jpeg'
            },
            {
                'scene_id': 4,
                'image_url': 'https://replicate.delivery/xezq/C5iNZpAMVNbhDFwuGIflCEIdVeI4jdSGwfv6ZyTTM56aDRnqA/tmpl7mommbh.jpeg'
            },
            {
                'scene_id': 5,
                'image_url': 'https://replicate.delivery/xezq/X2z9Gco8RQaUDRPiKxyD47aefy5re5bFmjbiC8Pf3pbxGiOVB/tmpby862ft9.jpeg'
            }
        ]
    },
    'model_name': 'kling-v2',
    'aspect_ratio': '9:16'
}

print('Testing the fixed video generation tool with actual error data...')
try:
    result = generate_scene_videos_tool.invoke(test_input)
    print('SUCCESS: Tool executed without error!')
    print('Result type:', type(result))
    print('Result preview:', result[:500] + '...' if len(result) > 500 else result)
except Exception as e:
    print('ERROR:', str(e))
    import traceback
    traceback.print_exc()