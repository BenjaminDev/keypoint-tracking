from typing import List

from numpy.core.fromnumeric import ndim

from dask_image.imread import imread
import napari
import numpy as np
from magicgui.widgets import ComboBox, Container
from napari.layers.points import Points
import sys
sys.path.append(".")
from utils import COLORS, Keypoints
COLOR_CYCLE = [
        '#1f77b4',
        '#ff7f0e',
        '#2ca02c',
        '#d62728',
        '#9467bd',
        '#8c564b',
        '#e377c2',
        '#7f7f7f',
        '#bcbd22',
        '#17becf'
]
COLOR_CYCLE = [COLORS[o.lower()] for o in Keypoints._fields]



def create_label_menu(points_layer, labels):
    """Create a label menu widget that can be added to the napari viewer dock

    Parameters:
    -----------
    points_layer : napari.layers.Points
        a napari points layer
    labels : List[str]
        list of the labels for each keypoint to be annotated (e.g., the body parts to be labeled).

    Returns:
    --------
    label_menu : QComboBox
        the label menu qt widget
    """
    # Create the label selection menu
    label_menu = ComboBox(label='feature_label', choices=labels)
    label_widget = Container(widgets=[label_menu])


    def update_label_menu(event):
        """Update the label menu when the point selection changes"""
        new_label = str(points_layer.current_properties['label'][0])
        if new_label != label_menu.value:
            label_menu.value = new_label

    points_layer.events.current_properties.connect(update_label_menu)

    def label_changed(event):
        """Update the Points layer when the label menu selection changes"""
        selected_label = event.value
        current_properties = points_layer.current_properties
        current_properties['label'] = np.asarray([selected_label])
        points_layer.current_properties = current_properties

    label_menu.changed.connect(label_changed)

    return label_widget


im_path = '/Volumes/external/wf/data/ARSession/frames/*.png'
labels = list(Keypoints._fields)
labels.append("END")
"""Create a GUI for annotating points in a series of images.

Parameters
----------
im_path : str
    glob-like string for the images to be labeled.
labels : List[str]
    list of the labels for each keypoint to be annotated (e.g., the body parts to be labeled).
"""
# viewer = napari.Viewer()
stack = imread(im_path)
# with napari.run():
viewer = napari.view_image(stack)
points_layer = viewer.add_points(
    data=np.empty((0, 3)),
    ndim=3,
    properties={'label': labels},
    edge_color='label',

    edge_color_cycle=COLOR_CYCLE,
    symbol='o',
    face_color='transparent',
    edge_width=8,
    size=12,
)
points_layer.edge_color_mode = 'cycle'

# add the label menu widget to the viewer
label_widget = create_label_menu(points_layer, labels)
viewer.window.add_dock_widget(label_widget)

@viewer.bind_key('.')
def next_label(event=None):
    """Keybinding to advance to the next label with wraparound"""

    current_properties = points_layer.current_properties
    current_label = current_properties['label'][0]

    ind = list(labels).index(current_label)
    new_ind = (ind + 1) % len(labels)
    new_label = labels[new_ind]
    try:
        current_properties['label'] = np.array([new_label])
        points_layer.current_properties = current_properties
    except:
        pass

def next_on_click(layer, event):
    """Mouse click binding to advance the label when a point is added"""
    if layer.mode == 'add':
        print("trying to go to next label")
        next_label()
        print("got there")
        # by default, napari selects the point that was just added
        # disable that behavior, as the highlight gets in the way
        layer.selected_data = {}

points_layer.mode = 'add'
points_layer.mouse_drag_callbacks.append(next_on_click)

@viewer.bind_key(',')
def prev_label(event):
    """Keybinding to decrement to the previous label with wraparound"""
    current_properties = points_layer.current_properties
    current_label = current_properties['label'][0]
    ind = list(labels).index(current_label)
    n_labels = len(labels)
    new_ind = ((ind - 1) + n_labels) % n_labels
    new_label = labels[new_ind]
    current_properties['label'] = np.array([new_label])
    points_layer.current_properties = current_properties

@viewer.bind_key('q')
def next_image(event=None):
    """"""
    # breakpoint()
    image_idx,y,z = viewer.dims.current_step
    viewer.dims.current_step = (image_idx+1, y,z)
    current_properties = points_layer.current_properties
    current_properties['label'] = np.array([labels[0]])
    points_layer.current_properties = current_properties
    # data = viewer.layers[1].data
    # previous_frame_points = data[data[:,0]==data[-1, 0]]
    # previous_frame_points[:,0]+=1
    # labels = viewer.layers[1].properties["label"]
    # previous_frame_labels = labels[-len(previous_frame_points):]

    # labels = viewer.layers[1].properties["label"] = np.concatenate((labels, previous_frame_labels))
    # viewer.layers[1].data = np.concatenate((viewer.layers[1].data,data))



@viewer.bind_key('f')
def save(event):
    # breakpoint()
    """Keybinding to decrement to the previous label with wraparound"""
    # current_properties = points_layer.current_properties
    # current_label = current_properties['label'][0]
    # ind = list(labels).index(current_label)
    # n_labels = len(labels)
    # new_ind = ((ind - 1) + n_labels) % n_labels
    # new_label = labels[new_ind]
    # current_properties['label'] = np.array([new_label])
    # points_layer.current_properties = current_properties
    print("sss")

    points_layer.save('file.csv')
print("gasdjklfglajksgfjkas")
napari.run()