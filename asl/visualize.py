from preprocess import get_char_dict
import plotly.graph_objects as go

char_dict = get_char_dict()
inv_dict = {v: k for k, v in char_dict.items()}


def visualise2d_landmarks(coordinates, title):
    fig = go.Figure()
    for i in range(len(coordinates)):
        frame = coordinates[i, :, :]
        frame = frame[(frame[:, 0] > -100)]
        if len(frame) == 0:
            continue
        trace = go.Scatter(
            # visible=False,
            x=frame[:, 0],
            y=frame[:, 1],
            mode="markers",
            marker=dict(size=9),
        )
        fig.add_trace(trace)
    fig.data[0].visible = True
    steps = []
    for i in range(len(fig.data)):
        step = dict(
            method="update",
            args=[
                {"visible": [False] * len(fig.data)},
            ],
        )
        step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
        steps.append(step)

    sliders = [dict(active=0, steps=steps)]

    fig.update_layout(sliders=sliders)
    fig.update_layout(
        title=title,
        xaxis=dict(range=[-10, 10], autorange=False, title="x", scaleanchor="y"),
        yaxis=dict(range=[10, -5], autorange=False, title="y"),
    )
    fig.show()


def visualize_train(sequence_id, coordinates, label_code):
    label = [inv_dict[x] for x in label_code]
    label = [x for x in label if len(x) == 1]
    label = "".join(label)
    title = f"sequence {sequence_id} {label}"
    visualise2d_landmarks(coordinates, title)
