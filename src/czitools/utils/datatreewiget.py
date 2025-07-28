# -*- coding: utf-8 -*-

##################################################################################################
# File: datatreewidget.py
# Author: sebi06
#
# Disclaimer: This code is purely experimental. Feel free to
# use it at your own risk.
#
# Adapted from: https://pyqtgraph.readthedocs.io/en/latest/_modules/pyqtgraph/widgets/DataTreeWidget.html#DataTreeWidget
#
#######################################################################################################################

from qtpy.QtWidgets import (
    QTreeWidget,
    QTreeWidgetItem,
    QPlainTextEdit,
)


# from PyQt5.QtWidgets import (
#     QTreeWidget,
#     QTreeWidgetItem,
#     QPlainTextEdit,
# )

from pyqtgraph import TableWidget
import numpy as np
import traceback
import types
from collections import OrderedDict


class DataTreeWidget(QTreeWidget):
    """
    Widget for displaying hierarchical python data structures
    (eg, nested dicts, lists, and arrays)
    """

    def __init__(self, parent=None, data=None):
        QTreeWidget.__init__(self, parent)
        self.setVerticalScrollMode(self.ScrollMode.ScrollPerPixel)
        self.setData(data)
        self.setColumnCount(3)
        self.setHeaderLabels(["Parameter", "Value", "Type"])
        self.setAlternatingRowColors(True)

    def setData(self, data, expandlevel=0, hideRoot=False):
        """data should be a dictionary."""
        self.clear()
        self.widgets = []
        self.nodes = {}
        self.buildTree(data, self.invisibleRootItem(), hideRoot=hideRoot)
        self.expandToDepth(expandlevel)
        self.resizeColumnToContents(0)
        # self.resizeColumnToContents(1)

    def buildTree(self, data, parent, name="", hideRoot=False, path=()):
        if hideRoot:
            node = parent
        else:
            node = QTreeWidgetItem([name, "", ""])
            parent.addChild(node)

        # record the path to the node so it can be retrieved later
        # (this is used by DiffTreeWidget)
        self.nodes[path] = node

        typeStr, desc, childs, widget = self.parse(data)

        # Truncate description and add text box if needed
        if len(desc) > 100:
            desc = desc[:97] + "..."
            if widget is None:
                widget = QPlainTextEdit(str(data))
                widget.setMaximumHeight(200)
                widget.setReadOnly(True)

        node.setText(1, desc)
        node.setText(2, typeStr)

        # Add widget to new subnode
        if widget is not None:
            self.widgets.append(widget)
            subnode = QTreeWidgetItem(["", "", ""])
            node.addChild(subnode)
            self.setItemWidget(subnode, 0, widget)
            subnode.setFirstColumnSpanned(True)

        # recurse to children
        for key, data in childs.items():
            self.buildTree(data, node, str(key), path=path + (key,))

    def parse(self, data):
        """
        Given any python object, return:
          * type
          * a short string representation
          * a dict of sub-objects to be parsed
          * optional widget to display as sub-node
        """
        # defaults for all objects
        typeStr = type(data).__name__
        if typeStr == "instance":
            typeStr += ": " + data.__class__.__name__
        widget = None
        desc = ""
        childs = {}

        # type-specific changes
        if isinstance(data, dict):
            desc = "length=%d" % len(data)
            if isinstance(data, OrderedDict):
                childs = data
            else:
                try:
                    childs = OrderedDict(sorted(data.items()))
                except TypeError:  # if sorting falls
                    childs = OrderedDict(data.items())

        elif isinstance(data, (list, tuple)):
            desc = "length=%d" % len(data)
            childs = OrderedDict(enumerate(data))

        elif isinstance(data, np.ndarray):
            desc = "shape=%s dtype=%s" % (data.shape, data.dtype)
            table = TableWidget()
            table.setData(data)
            table.setMaximumHeight(200)
            widget = table

        elif isinstance(data, types.TracebackType):  ## convert traceback to a list of strings
            frames = list(map(str.strip, traceback.format_list(traceback.extract_tb(data))))

            widget = QPlainTextEdit("\n".join(frames))
            widget.setMaximumHeight(200)
            widget.setReadOnly(True)
        else:
            desc = str(data)

        return typeStr, desc, childs, widget
