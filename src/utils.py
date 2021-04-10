import plotly.graph_objects as go

def plot_3d(x, y, z, classes=None, fig=None, row=None, col=None):

    if fig is None:
        fig = go.Figure()

    fig.add_trace(
        go.Scatter3d(x=x, y=y, z=z, mode="markers", marker=dict(color=classes, size=4)),
        row=row,
        col=col
    )

    return fig