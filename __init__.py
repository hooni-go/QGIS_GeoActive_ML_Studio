# -*- coding: utf-8 -*-
"""
This script initializes the plugin, making it known to QGIS.
"""

def classFactory(iface):
    """Load QGISGeoActiveMLStudioPlugin class from file plugin.
    
    :param iface: A QGIS interface instance.
    :type iface: QgsInterface
    """
    from .plugin import QGISGeoActiveMLStudioPlugin
    return QGISGeoActiveMLStudioPlugin(iface)
