import meshio

domain = 't-d5'


def formating(domain):
    # filename = f"D:\\CAD\\gmsh mesh\\freemesh\\D4-D10\\{domain}.inp" #\\gmsh
    filename = f"D:\CAD\\tnnls\gmsh\\{domain}.inp"
    # filename = f"D:\\CAD\\gmsh mesh\\{domain}.inp"
    m = meshio.Mesh.read(filename, "abaqus")  # same arguments as meshio.read
    # m.write(f"D:\\CAD\\gmsh mesh\\freemesh\\D4-D10\\{domain}.vtk") #\\gmsh
    m.write(f"D:\\CAD\\tnnls\gmsh\\{domain}.vtk")  # \\gmsh

#
# domains = [
#     # 't-t5-2',
#     # 't-t2-2',
#     # 't-t3-2',
#     # 't-t4-2'
#     # 't-d5',
#     # 't-d5-2',
#     # 'g-d5',
#     # 'g-d5-2',
#     # 't-d6',
#     # 't-d6-2',
#     # 'g-d6',
#     # 'g-d6-2',
#     # 't-d7',
#     # 't-d7-2',
#     # 'g-d7',
#     # 'g-d7-2',
#     # 't-d8',
#     # 't-d8-2',
#     # 'g-d8',
#     # 'g-d8-2',
#     # 't-d9',
#     # 't-d9-2',
#     # 'g-d9',
#     # 'g-d9-2',
#     # 'freemesh-d4',
#     # 'freemesh-d5',
#     # 'freemesh-d6',
#     # 'freemesh-d7',
#     # 'freemesh-d8',
#     # 'freemesh-d9',
#     # 'freemesh-d10',
#     # 'g_random',
#     # 'g_fly',
#     # 'g_dragon',
#     # 'pave_fly',
#     # 'pave_random',
#     # 'pave_dragon',
#     # 'fly2',
#     # 'random2',
#     # 'dragon2'
#     'd0', 'd1', 'd2', 'd3', 'd4'
# ]
#
# for d in domains:
#    formating(d)
