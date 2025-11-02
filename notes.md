uv run -m video_viewer.analysis_platform

# TODO
MAJOR BUG: <br/>
in file_maanger, save_project: <br/>
json.dump(project_data, f, cls=EnhancedJSONEncoder,indent=4)<br/>
if it fails (for example, I used type int64 which is not JSON serializable) it wipes out data not yet written to JSON.