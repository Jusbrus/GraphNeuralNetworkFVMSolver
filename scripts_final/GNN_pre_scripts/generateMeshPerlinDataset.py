import os
import copy
import numpy as np
import math

class GenerateMeshMoreObjects:

    def __init__(self, objects_list,box_list, variable_dict_variant,variable_dict_invariant, save_path):
        self.objects_list = objects_list
        self.box_list = box_list
        self.variable_dict_variant = variable_dict_variant
        self.variable_dict_invariant = variable_dict_invariant
        self.save_path = save_path

        self.get_variable_list()
        self.get_objects()
        self.get_boxes()
        self.get_curve_loops()
        self.get_plane_surfaces()
        self.get_transfinite_curves()
        self.get_physical_surfaces()

    def get_airfoil_coordinates(self,NACA_airfoil, n_nodes, chord, alpha):
        M = int(NACA_airfoil[0]) / 100
        P = int(NACA_airfoil[1]) / 10
        T = int(NACA_airfoil[2:]) / 100
        beta_tab = np.linspace(0, np.pi, math.ceil(n_nodes / 2))
        x_tab = (1 - np.cos(beta_tab)) / 2
        camber_tab = np.zeros(math.ceil(n_nodes / 2))
        gradient_tab = np.zeros(math.ceil(n_nodes / 2))
        thickness_tab = np.zeros(math.ceil(n_nodes / 2))
        x_upper = np.zeros(math.ceil(n_nodes / 2))
        x_lower = np.zeros(math.ceil(n_nodes / 2))
        y_upper = np.zeros(math.ceil(n_nodes / 2))
        y_lower = np.zeros(math.ceil(n_nodes / 2))

        for i in range(len(camber_tab)):
            thickness_tab[i] = (T / 0.2) * ((0.2969 * (x_tab[i] ** (0.5))) +
                                            (-0.126 * (x_tab[i])) +
                                            (-0.3516 * (x_tab[i] ** (2))) +
                                            (0.2843 * (x_tab[i] ** (3))) +
                                            (-0.1036 * (x_tab[i] ** (4))))
            if x_tab[i] < P:
                camber_tab[i] = (M / (P ** 2)) * (2 * P * x_tab[i] - x_tab[i] ** 2)
                gradient_tab[i] = (2 * M / (P ** 2)) * (P - x_tab[i])
            else:
                camber_tab[i] = (M / ((1 - P) ** 2)) * (1 - 2 * P + 2 * P * x_tab[i] - x_tab[i] ** 2)
                gradient_tab[i] = (2 * M / ((1 - P) ** 2)) * (P - x_tab[i])
            x_upper[i] = x_tab[i] - thickness_tab[i] * np.sin(np.arctan(gradient_tab[i]))
            x_lower[i] = x_tab[i] + thickness_tab[i] * np.sin(np.arctan(gradient_tab[i]))
            y_upper[i] = camber_tab[i] + thickness_tab[i] * np.cos(np.arctan(gradient_tab[i]))
            y_lower[i] = camber_tab[i] - thickness_tab[i] * np.cos(np.arctan(gradient_tab[i]))

        x_data = np.append(np.flip(x_lower), x_upper[1:])
        y_data = np.append(np.flip(y_lower), y_upper[1:])

        x_data = x_data * chord
        y_data = y_data * chord

        alpha_rad = np.radians(-alpha)
        cos_alpha = np.cos(alpha_rad)
        sin_alpha = np.sin(alpha_rad)

        x_shifted = x_data - (chord / 2)
        y_shifted = y_data

        x_rotated = cos_alpha * x_shifted - sin_alpha * y_shifted + (chord / 2)
        y_rotated = sin_alpha * x_shifted + cos_alpha * y_shifted

        return x_rotated[:-1], y_rotated[:-1]

    def get_airfoil_list(self,NACA_airfoil, n_nodes, chord,i_airfoil,alpha,x_leadingEdge,y_leadingEdge):
        x,y = self.get_airfoil_coordinates(NACA_airfoil, n_nodes+2, chord,alpha)
        x += x_leadingEdge
        y += y_leadingEdge

        script_one_airfoil = [f"// AIRFOIL BEGIN -----------------------------------------\n"]
        i = i_airfoil * 1000 + 1
        for j in range (len(x)):
            script_one_airfoil.append(f"Point({i}) = {{{x[j]}, {y[j]}, 0, 1}};\n")
            script_one_airfoil.append(f"//+\n")
            i += 1
        script_one_airfoil.append(f"\n")

        i = i_airfoil * 1000 + 1
        for j in range (len(x)):
            if j != len(x)-1:
                script_one_airfoil.append(f"Line({i}) = {{{i}, {i+1}}};\n")
            else:
                script_one_airfoil.append(f"Line({i}) = {{{i}, {i_airfoil * 1000 + 1}}};\n")

            script_one_airfoil.append(f"//+\n")
            i += 1

        script_one_airfoil.append(f"\n")

        script_one_airfoil.append("Curve Loop(" + str(i_airfoil) + ") = {" + ", ".join(str(i) for i in range(i_airfoil * 1000 + 1,i_airfoil * 1000 + 1+len(x))) + "};\n")
        script_one_airfoil.append(f"// AIRFOIL END -----------------------------------------\n")
        script_one_airfoil.append(f"\n")

        return script_one_airfoil

    def get_variable_list(self):
        self.script_variable_list = []
        for key in self.variable_dict_variant.keys():
            new_line = key + " = " + str(self.variable_dict_variant[key]) + ";\n"
            self.script_variable_list.append(new_line)

        self.script_variable_list.append(f"\n")

        # print(self.script_variable_list)

    def get_objects(self):
        self.script_objects = []
        i = 5
        j = 500

        for dict in self.objects_list:
            if dict["type"] == "circle":
                new_line = f"Circle({i}) = {{{dict['x']}, {dict['y']}, 0, {dict['r']}, 0, 2*Pi}};\n"
                self.script_objects.append(new_line)
            elif dict["type"] == "ellipse":
                new_line = f"Ellipse({i}) = {{{dict['x']}, {dict['y']}, 0, {dict['rx']}, {dict['ry']}, 0, 2*Pi}};\n"
                self.script_objects.append(new_line)
                new_line = f"Rotate {{{{0, 0, 1}}, {{{dict['x']}, {dict['y']}, 0}}, {dict['alpha'] * -np.pi/180}}} {{Curve{{{i}}};}}\n"
                self.script_objects.append(new_line)
            elif dict["type"] == "rectangle":
                dict['x1'] = dict["x"] - 0.5 * dict["w"]
                dict['x2'] = dict["x"] + 0.5 * dict["w"]
                dict['y1'] = dict["y"] - 0.5 * dict["h"]
                dict['y2'] = dict["y"] + 0.5 * dict["h"]

                tab = self.get_box(dict,j,i-3,alpha=dict["alpha"])
                j+=4
                self.script_objects += tab
            elif dict["type"] == "airfoil":
                dict["x"] -= 0.5 * dict["c"]
                tab = self.get_airfoil_list(dict["naca"], self.variable_dict_variant[dict["n"]], dict["c"], i - 3, dict["alpha"], dict["x"], dict["y"])
                self.script_objects += tab


            self.script_objects.append(f"//+\n")
            i += 1

        self.script_objects.append(f"\n")

        # Rectangle(221) = {0.5, 0.9, 0.1, 1, 0.5, 0};

        # print(self.script_objects)

    def get_box(self,dict,i,j,alpha=0):
        script_box = []
        x_tab = np.array([dict['x1'],dict['x1'],dict['x2'],dict['x2']])
        y_tab = np.array([dict['y1'],dict['y2'],dict['y2'],dict['y1']])

        x_center = (dict['x1'] + dict['x2']) / 2
        y_center = (dict['y1'] + dict['y2']) / 2

        alpha_rad = np.radians(-alpha)

        cos_alpha = np.cos(alpha_rad)
        sin_alpha = np.sin(alpha_rad)

        x_shifted = x_tab - x_center
        y_shifted = y_tab - y_center

        x_tab = cos_alpha * x_shifted - sin_alpha * y_shifted + x_center
        y_tab = sin_alpha * x_shifted + cos_alpha * y_shifted + y_center

        box_points = [f"// BOX {j - 1 - len(self.objects_list)} BEGIN -----------------------------------------\n",
                      f"Point({i + 1}) = {{{x_tab[0]}, {y_tab[0]}, 0}};\n",
                      f"//+\n",
                      f"Point({i + 2}) = {{{x_tab[1]}, {y_tab[1]}, 0}};\n",
                      f"//+\n",
                      f"Point({i + 3}) = {{{x_tab[2]}, {y_tab[2]}, 0}};\n",
                      f"//+\n",
                      f"Point({i + 4}) = {{{x_tab[3]}, {y_tab[3]}, 0}};\n",
                      f"//+\n",
                      f"\n"]

        box_lines = [f"Line({i + 1}) = {{{i + 1}, {i + 2}}};\n",
                     f"//+\n",
                     f"Line({i + 2}) = {{{i + 2}, {i + 3}}};\n",
                     f"//+\n",
                     f"Line({i + 3}) = {{{i + 3}, {i + 4}}};\n",
                     f"//+\n",
                     f"Line({i + 4}) = {{{i + 4}, {i + 1}}};\n",
                     f"//+\n",
                     f"\n"]

        curve_loop = ["Curve Loop(" + str(j) + ") = {" + ", ".join(str(i) for i in range(i + 1, 5 + i)) + "};\n",
                      f"\n"]

        transfinite_curves = [
            f'Transfinite Curve {{{i + 1}}} = {dict["ny"]} Using Progression 1;\n',
            f"//+\n",
            f'Transfinite Curve {{{i + 2}}} = {dict["nx"]} Using Progression 1;\n',
            f"//+\n",
            f'Transfinite Curve {{{i + 3}}} = {dict["ny"]} Using Progression 1;\n',
            f"//+\n",
            f'Transfinite Curve {{{i + 4}}} = {dict["nx"]} Using Progression 1;\n',
            f"//+\n",
            f"// BOX {j - 1 - len(self.objects_list)} END -------------------------------------------\n", ]

        script_box += box_points
        script_box += box_lines
        script_box += curve_loop
        script_box += transfinite_curves

        return script_box

    def get_boxes(self):
        self.script_boxes = []
        j = 2 + len(self.objects_list)
        i = 100

        for dict in self.box_list:

            self.script_boxes += self.get_box(dict,i,j)

            i+=4
            j+=1

            # ["Plane Surface(1) = {" + ", ".join(str(i) for i in plane_list_main) + "};\n"]

        self.script_boxes.append(f"\n")


        # print(self.script_objects)

    def get_curve_loops(self):
        self.script_curve_loops = [f"Curve Loop(1) = {{1, 2, 3, 4}};\n",
                                   f"//+\n"]

        for i in range(len(self.objects_list)):
            if self.objects_list[i]["type"] in ["circle","ellipse"]:
                new_line = "Curve Loop(" + str(i + 2) + ") = {" + str(i + 5) + "};\n"
                self.script_curve_loops.append(new_line)
                self.script_curve_loops.append(f"//+\n")
        self.script_curve_loops.append(f"\n")

        # print(self.script_curve_loops)

    def get_plane_surfaces(self):
        plane_list_main = [1]
        i = 2

        extra_planes_list = []
        for dict in self.objects_list:
            if dict["box"] == False:
                plane_list_main.append(i)
            i += 1

        for dict in self.box_list:
            plane_list_main.append(i)
            sub_planes_list = [i] + list(np.array(dict["objects"])+2)
            extra_planes_list.append(sub_planes_list)

            i += 1

        # self.script_plane_surfaces = ["Plane Surface(1) = {" + ", ".join(str(i) for i in range(1, len(objects_list) + 2)) + "};\n"]
        self.script_plane_surfaces = ["Plane Surface(1) = {" + ", ".join(str(i) for i in plane_list_main) + "};\n"]
        self.script_plane_surfaces.append(f"//+\n")

        i = 2
        for tab in extra_planes_list:
            new_line = f"Plane Surface({i}) = {{" + ", ".join(str(i) for i in tab) + "};\n"
            i += 1
            self.script_plane_surfaces.append(new_line)
            self.script_plane_surfaces.append(f"//+\n")

        self.script_plane_surfaces.append(f"\n")


        # print(self.script_plane_surfaces)

    def get_transfinite_curves(self):
        # type_list = ["Bump", "Progression", "Progression", "Progression"]
        # value_list = [0.2,1,0.8,1]
        #
        # type_list = ["Progression", "Progression", "Progression", "Progression"]
        # value_list = [1,1,1,1]
        self.script_transfinite_curves = [f'Transfinite Curve {{1}} = ny Using  {self.variable_dict_invariant["cell_distribution_type"][0]} {self.variable_dict_invariant["cell_distribution_value"][0]};\n',
                        f"//+\n",
                        f'Transfinite Curve {{2}} = nx Using {self.variable_dict_invariant["cell_distribution_type"][1]} {self.variable_dict_invariant["cell_distribution_value"][1]};\n',
                        f"//+\n",
                        f'Transfinite Curve {{3}} = ny Using {self.variable_dict_invariant["cell_distribution_type"][2]} {self.variable_dict_invariant["cell_distribution_value"][2]};\n',
                        f"//+\n",
                        f'Transfinite Curve {{4}} = nx Using {self.variable_dict_invariant["cell_distribution_type"][3]} {self.variable_dict_invariant["cell_distribution_value"][3]};\n',
                        f"//+\n"]

        for i in range(len(self.objects_list)):
            new_line = "Transfinite Curve {" + str(i + 5) + "} = " + str(
                self.objects_list[i]["n"]) + " Using Progression 1;\n"

            self.script_transfinite_curves.append(new_line)
            self.script_transfinite_curves.append(f"//+\n")

        self.script_transfinite_curves.append(f"\n")

    def get_physical_surfaces(self):
        side_index_list = [6 + len(self.objects_list) + 4 * len(self.box_list)]
        object_index_list = np.arange(6,6+len(self.objects_list))
        for dict in self.box_list:
            i = 1
            for i_object in dict["objects"]:
                object_index_list[i_object] = side_index_list[-1]+i
                i+=1
            side_index_list.append(side_index_list[-1]+i)

        if len(self.objects_list)>0:
            self.script_physical_surfaces = [f'Physical Volume("Fluid", 1) = {{{", ".join(str(i) for i in range(1, 2 + len(self.box_list)))}}};\n',
                            f"//+\n",
                            f'Physical Surface("Left", 2) = {{3}};\n',
                            f"//+\n",
                            f'Physical Surface("Lower", 3) = {{6}};\n',
                            f"//+\n",
                            f'Physical Surface("Right", 4) = {{5}};\n',
                            f"//+\n",
                            f'Physical Surface("Upper", 5) = {{4}};\n',
                            f"//+\n",
                            f'Physical Surface("Side", 6) = {{{", ".join(str(i) for i in range(1, 2 + len(self.box_list)))}, {", ".join(str(i) for i in side_index_list)}}};\n',
                            f"//+\n"]
        else:
            self.script_physical_surfaces = [f'Physical Volume("Fluid", 1) = {{1}};\n',
                            f"//+\n",
                            f'Physical Surface("Left", 2) = {{2}};\n',
                            f"//+\n",
                            f'Physical Surface("Lower", 3) = {{5}};\n',
                            f"//+\n",
                            f'Physical Surface("Right", 4) = {{4}};\n',
                            f"//+\n",
                            f'Physical Surface("Upper", 5) = {{3}};\n',
                            f"//+\n",
                            f'Physical Surface("Side", 6) = {{1,6}};\n',
                            f"//+\n"]

        for i in range(len(self.objects_list)):
            new_line = f'Physical Surface("Object", {7+i}) = {{{object_index_list[i]}}};\n'

            self.script_physical_surfaces.append(new_line)
            self.script_physical_surfaces.append(f"//+\n")

        # print(self.script_physical_surfaces)



    def get_mesh(self):

        script_start = [f"// Gmsh project created on Sun May 19 18:32:31 2024\n",
                        f'SetFactory("OpenCASCADE");\n',
                        f"\n"]

        script_points = [f"Point(1) = {{0, 0, 0}};\n",
                        f"//+\n",
                        f"Point(2) = {{0, {self.variable_dict_invariant['y']}, 0}};\n",
                        f"//+\n",
                        f"Point(3) = {{{self.variable_dict_invariant['x']}, {self.variable_dict_invariant['y']}, 0}};\n",
                        f"//+\n",
                        f"Point(4) = {{{self.variable_dict_invariant['x']}, 0, 0}};\n",
                        f"//+\n",
                        f"\n"]

        script_lines = [f"Line(1) = {{1, 2}};\n",
                        f"//+\n",
                        f"Line(2) = {{2, 3}};\n",
                        f"//+\n",
                        f"Line(3) = {{3, 4}};\n",
                        f"//+\n",
                        f"Line(4) = {{4, 1}};\n",
                        f"//+\n",
                        f"\n"]

        extrude_list = [f"Extrude {{0, 0, 1}} {{\n",
                        f'Surface{{{", ".join(str(i) for i in np.arange(1,2+len(self.box_list)))}}}; Layers {{1}}; Recombine;\n',
                        f"}}\n",
                        f"//+\n",
                        f"\n"]

        end_list = [f"Mesh.Algorithm = 1;\n"]


        script = (script_start + self.script_variable_list +
                  script_points + script_lines + self.script_objects +
                  self.script_curve_loops + self.script_boxes + self.script_plane_surfaces +
                  self.script_transfinite_curves + extrude_list +
                  self.script_physical_surfaces + end_list)
        # print(script)

        with open(self.save_path, 'w') as new_file:
            new_file.writelines(script)



if __name__ == '__main__':

    variable_dict_variant = {"nx": 24,
                             "ny": 24,
                            "ncyl": 200,
                            "nx_box": 16,
                            "ny_box": 16,
                            "nx_rect": 50,
                            "ny_rect": 50,
                             }

    variable_dict_variant = {"nx": 8,
                             "ny": 8,
                            "ncyl": 7,
                            "nx_box": 4,
                            "ny_box": 4,
                             }

    variable_dict_invariant = {
                            "y": 16,
                            "x": 16,
                            "cell_distribution_type": ["Progression", "Progression", "Progression", "Progression"],
                            "cell_distribution_value": [1, 1, 1, 1],
                     }

    # variable_dict_invariant = {
    #                         "y": 2,
    #                         "x": 1,
    #                  }

    # objects_list = [{"type":"circle", "x":0.4, "y": 1.5, "r": 0.1, "n":"ncyl","box": False},
    #                 {"type":"ellipse", "x":0.3, "y": 0.7, "rx": 0.2, "ry":0.1, "alpha":10, "n":"ncyl","box": False},
    #                 {"type":"circle", "x":0.8, "y": 0.5, "r": 0.1, "n":"ncyl","box": False}]

    # objects_list = [{"type":"circle", "x":0.4, "y": 1, "r": 0.1, "n":"ncyl","box": True},
    #                 {"type":"ellipse", "x":0.3, "y": 0.65, "rx": 0.2, "ry":0.02,"alpha":10, "n":"ncyl","box": True},
    #                 {"type":"circle", "x":0.8, "y": 0.5, "r": 0.1, "n":"ncyl","box": False}]


    # objects_list = [{"type":"airfoil","naca": "4735", "x":0.3, "y": 1, "c": 0.2,"alpha":15, "n":"ncyl","box": True},
    #                 {"type":"ellipse", "x":0.3, "y": 0.65, "rx": 0.2, "ry":0.02,"alpha":10, "n":"ncyl","box": True},
    #                 {"type":"circle", "x":0.8, "y": 0.5, "r": 0.1, "n":"ncyl","box": False},
    #                 {"type": "circle", "x": 0.8, "y": 1.5, "r": 0.1, "n": "ncyl", "box": False}
    #                 ]

    # objects_list = [{"type":"airfoil","naca": "4735", "x":0.3, "y": 1, "c": 0.2,"alpha":15, "n":"ncyl","box": True},
    #                 {"type":"ellipse", "x":0.3, "y": 0.65, "rx": 0.2, "ry":0.02,"alpha":10, "n":"ncyl","box": True},
    #                 {"type":"circle", "x":0.8, "y": 0.5, "r": 0.1, "n":"ncyl","box": False},]

    objects_list = [{"type":"airfoil","naca": "4735", "x":0.3, "y": 1, "c": 0.2,"alpha":15, "n":"ncyl","box": True},
                    {"type":"ellipse", "x":0.3, "y": 0.65, "rx": 0.2, "ry":0.02,"alpha":10, "n":"ncyl","box": True},
                    {"type":"circle", "x":0.8, "y": 0.5, "r": 0.1, "n":"ncyl","box": False},
                    {"type": "rectangle", "x": 0.75, "y": 1.6, "w": 0.2, "h": 0.2, "nx": "nx_rect", "ny": "ny_rect",
                     "alpha": 30, "n": "ncyl", "box": False},
                    ]

    objects_list = [{"type":"circle", "x":8, "y": 8, "r": 0.5, "n":"ncyl","box": True},
                    {"type":"airfoil","naca": "2714", "x":10, "y": 8, "c": 1,"alpha":5, "n":"ncyl","box": True}]
    # objects_list = [{"type":"airfoil","naca": "2714", "x":8, "y": 8, "c": 1,"alpha":5, "n":"ncyl","box": True}]

    objects_list = [{"type":"circle", "x":8, "y": 8, "r": 0.5, "n":"ncyl","box": True}]
    objects_list = [{"type":"ellipse", "x":8, "y": 8, "rx": 0.5, "ry":0.05,"alpha":10, "n":"ncyl","box": True}]
    #
    # box_list = [{"x1": 7, "x2": 8.75, "y1": 7, "y2": 9, "nx": "nx_box1", "ny": "ny_box1", "objects": [0]},
    #             {"x1": 9, "x2": 26, "y1": 7, "y2": 9, "nx": "nx_box2", "ny": "ny_box2", "objects": [1]},
    #                 ]

    box_list = [{"x1": 7, "x2": 9, "y1": 7, "y2": 9, "nx": "nx_box", "ny": "ny_box", "objects": [0]},
                    ]

    box_list = []
    objects_list = []

    save_path = "/home/justinbrusche/dataVaryingMeshes/geo_file_random.geo"

    a = GenerateMeshMoreObjects(objects_list,box_list, variable_dict_variant,variable_dict_invariant, save_path)
    a.get_mesh()

    # print(f"Point(1) = {{{a}, {b}, {c}}};\n")


