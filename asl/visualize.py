from .constants import Constants
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np

pio.templates.default = "simple_white"


def visualise2d_landmarks(coordinates, title):
    connections = [
        [
            0,
            1,
            2,
            3,
            4,
        ],
        [0, 5, 6, 7, 8],
        [0, 9, 10, 11, 12],
        [0, 13, 14, 15, 16],
        [0, 17, 18, 19, 20],
    ]
    # coordinates (T,F,2)
    coordinates = coordinates[coordinates[:, 0, 0] != Constants.INPUT_PAD, :, :]
    frames_l = []
    for i in range(len(coordinates)):
        frame = coordinates[i, :, :]
        colors = ["green"] * len(frame)
        for j in Constants.LANDMARK_INDICES["LHAND"]:
            colors[j] = "blue"
        for j in Constants.LANDMARK_INDICES["RHAND"]:
            colors[j] = "red"
        if len(frame) == 0:
            continue
        if np.sum(np.abs(frame)) == 0:
            continue

        traces = [
            go.Scatter(
                x=frame[:, 0],
                y=frame[:, 1],
                mode="markers",
                marker=dict(color=colors, size=9),
            )
        ]
        for hand in ["RHAND", "LHAND"]:
            for seg in connections:
                iconnect = [Constants.LANDMARK_INDICES[hand][i] for i in seg]
                trace = go.Scatter(
                    x=frame[iconnect, 0],
                    y=frame[iconnect, 1],
                    mode="lines",
                    line=dict(color="black", width=2),
                )
                traces.append(trace)

        if i == 0:
            first_traces = traces

        frame_data = go.Frame(
            data=traces,
        )
        frames_l.append(frame_data)

    fig = go.Figure(data=first_traces, frames=frames_l)

    fig.update_layout(
        width=500,
        height=800,
        scene={
            "aspectmode": "data",
        },
        updatemenus=[
            {
                "buttons": [
                    {
                        "args": [
                            None,
                            {
                                "frame": {"duration": 200, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                            },
                        ],
                        "label": "&#9654;",  # Play button symbol
                        "method": "animate",
                    },
                    {
                        "args": [
                            [None],
                            {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"},
                        ],
                        "label": "&#10074;&#10074;",  # Pause button symbol
                        "method": "animate",
                    },
                ],
                "direction": "left",
                "pad": {"r": 100, "t": 100},
                "font": {"size": 30},
                "type": "buttons",
                "x": 0.1,
                "y": 0,
            }
        ],
    )
    camera = dict(up=dict(x=0, y=-1, z=0), eye=dict(x=0, y=0, z=2.5))
    fig.update_layout(title_text=title, title_x=0.5)
    fig.update_layout(scene_camera=camera, showlegend=False)
    # fig.update_layout(
    #    xaxis=dict(visible=False),
    #    yaxis=dict(visible=False),
    # )
    fig.update_yaxes(autorange="reversed")

    fig.show()


def visualize_train(sequence_id, coordinates, label_code):
    label = [Constants.inv_dict[x] for x in label_code]
    label = [x for x in label if len(x) == 1]
    label = "".join(label)
    title = f"sequence {sequence_id} Phrase: {label}"
    visualise2d_landmarks(coordinates, title)
