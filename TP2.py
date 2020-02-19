import halfedge_mesh
from halfedge_mesh import get_geodesic_distance
from halfedge_mesh import export_geodesic_distance
from halfedge_mesh import color_connected_components
from halfedge_mesh import meshSegmentationBySize
from halfedge_mesh import simpleMeshSegmentationBySize

# TODO : comments, color, faces, line break,

# .off are supported

mesh = halfedge_mesh.HalfedgeMesh("bonhomme.off")

# Returns a list of Vertex type (in order of file)--similarly for halfedges,
# and facets

mesh.vertices

# The number of facets in the mesh
len(mesh.facets)

# Get the 10th halfedge

mesh.halfedges[10]
# Get the halfedge that starts at vertex 25 and ends at vertex 50
#mesh.get_halfedge(25, 50)


a = mesh.vertices[1]
b = mesh.vertices[75]

#color_connected_components(mesh)
#export_geodesic_distance(mesh, a, b)
meshSegmentationBySize(mesh)