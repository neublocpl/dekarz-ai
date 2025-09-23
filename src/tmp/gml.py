from lxml import etree
import numpy as np
import pyvista as pv

# Namespace CityGML
ns = {
    "gml": "http://www.opengis.net/gml",
    "bldg": "http://www.opengis.net/citygml/building/2.0"
}
path = '/Users/grzegorzstermedia/Downloads/Modele_3D/1433_N-34-116-D-c-4-4.gml'
tree = etree.parse(path)
root = tree.getroot()

# Kolory dla typów powierzchni
colors = {
    "GroundSurface": "green",
    "WallSurface": "lightgray",
    "RoofSurface": "red"
}
plotter = pv.Plotter()


for surf_type in ["GroundSurface", "WallSurface", "RoofSurface"]:
    for surface in root.xpath(f"//bldg:{surf_type}", namespaces=ns):
        # zbieranie wszystkich polygonów w powierzchni
        polygons = []
        for pg in surface.xpath(".//gml:Polygon", namespaces=ns):
            points = []
            for pos in pg.xpath(".//gml:pos", namespaces=ns):
                xyz = list(map(float, pos.text.split()))
                points.append(xyz)
            if len(points) > 2:
                polygons.append(np.array(points))
        # tworzenie PolyData dla każdego polygonu
        for poly in polygons:
            mesh = pv.PolyData(poly)
            # jeśli polygon ma więcej niż 3 punkty, triangulacja
            if len(poly) > 3:
                mesh = mesh.delaunay_2d()
            plotter.add_mesh(mesh, color=colors[surf_type], show_edges=True, opacity=0.8)

plotter.show()