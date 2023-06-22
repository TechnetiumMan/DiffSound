import plotly.graph_objs as go
from ipywidgets import interactive, IntSlider, HBox, VBox, Box, Layout, Text, Button, Output, FloatSlider
import numpy as np
import IPython
row_layout = Layout(
    display='flex',
    flex_flow='row',
    justify_content='space-around'
)
col_layout = Layout(
    display='flex',
    flex_flow='column',
    align_items='stretch',
    flex='1 1 auto'
)


class viewer():
    def __init__(self, vertices, elements, data=None, show_axis=False, title='', intensitymode='cell'):
        '''
        Shape of parameters:
        vertices    [node_num, 3]
        elements    [triangle_num, 3]
        data        [feature_num, triangle_num] or [feature_num, node_num]
        intensitymode   'cell' or 'vertex'
        '''
        self.title = title
        self.show_axis = show_axis
        self.vertices, self.elements, self.data = vertices.T, elements.T, data
        self.intensitymode = intensitymode
        self._in_batch_mode = False
        if self.data is None:
            self.items = [self.init_3D()]
        else:
            self.items = [
                self.init_data_selector(),
                self.init_3D()
            ]

    @property
    def box(self):
        return Box(
            self.items,
            layout=col_layout
        )

    def show(self):
        IPython.display.display(self.box)

    def clear(self):
        self.fig.data[0].x = []
        self.fig.data[0].y = []
        self.fig.data[0].z = []
        self.fig.data[0].i = []
        self.fig.data[0].j = []
        self.fig.data[0].k = []
        self.fig.data[0].intensity = []

    def init_3D(self):
        bound_max = self.vertices.max()
        bound_min = self.vertices.min()
        if self.data is None:
            self.fig = go.FigureWidget(data=[
                go.Mesh3d(
                    x=self.vertices[0],
                    y=self.vertices[1],
                    z=self.vertices[2],
                    # intensity=self.data[0],
                    # intensitymode='cell',
                    # colorscale='Jet',
                    i=self.elements[0],
                    j=self.elements[1],
                    k=self.elements[2],
                    showlegend=self.show_axis,
                    showscale=True,
                )
            ]
            )
        else:
            self.fig = go.FigureWidget(data=[
                go.Mesh3d(
                    x=self.vertices[0],
                    y=self.vertices[1],
                    z=self.vertices[2],
                    intensity=self.data[0],
                    intensitymode=self.intensitymode,
                    colorscale='Jet',
                    i=self.elements[0],
                    j=self.elements[1],
                    k=self.elements[2],
                    showlegend=self.show_axis,
                    showscale=True,
                )
            ]
            )
        self.fig.layout.height = 500
        self.fig.layout.width = 1000
        self.fig.layout.autosize = False
        self.fig.layout.title = self.title
        self.fig.update_layout(
            margin=dict(l=0, r=0, b=0, t=0, pad=0),
            title={'y': 0.9, 'x': 0.4},
        )
        self.fig.update_layout(scene_aspectmode='cube')
        if self.show_axis:
            scene = dict(
                xaxis=dict(range=[bound_min, bound_max]),
                yaxis=dict(range=[bound_min, bound_max]),
                zaxis=dict(range=[bound_min, bound_max]),
            )
        else:
            scene = dict(
                xaxis=dict(showticklabels=False, visible=False,
                           range=[bound_min, bound_max]),
                yaxis=dict(showticklabels=False, visible=False,
                           range=[bound_min, bound_max]),
                zaxis=dict(showticklabels=False, visible=False,
                           range=[bound_min, bound_max]),
            )
        self.fig.update_layout(scene=scene)
        self.fig.data[0].lighting = {
            "ambient": 0.7,
            "diffuse": 1,
            "specular": 0.3,
            "roughness": .5,
        }
        self.fig.update_layout(title_font_size=20)
        return Box([self.fig], layout=row_layout)

    def init_data_selector(self):
        self.int_range = IntSlider(min=0, max=len(
            self.data) - 1, step=1, value=0, layout=Layout(flex='3 1 auto'), description="Mode Index")

        def select(change):
            self.fig.data[0].intensity = self.data[self.int_range.value]
        self.int_range.observe(select, names='value')
        return Box([self.int_range], layout=row_layout)