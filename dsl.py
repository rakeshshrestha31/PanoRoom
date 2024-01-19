from ksecs.ECS.processors.pixel_processor import PixelProcessor
from ksecs.ECS.processors.entity_processor import EntityProcessor
from ksecs.ECS.processors.render_processor import RenderProcessor
from ksecs.ECS.processors.structure_processor import StructureProcessor


class EntityDsl(EntityProcessor):
    def process(self):
        for camera in self.shader.world.cameras:
            self.shader.world.delete_entity(camera)
        # radius_list = [0, 1000, 2000]
        pic_index = 0
        for room in self.shader.world.rooms:
            if room._id == '4430':
                height = 1200.0

                position = {
                    'x': room.position[0],
                    'y': room.position[1],
                    'z': height,
                }
                look_at = {
                    'x': room.position[0],
                    'y': room.position[1] + 500,
                    'z': height,
                }
                # for radius in radius_list:
                self.shader.world.add_camera(
                    id=str(pic_index),
                    cameraType="PANORAMA",
                    position=position,
                    lookAt=look_at,
                    up=[0, 0, 1],
                    imageWidth=1024,
                    imageHeight=512,
                )
                pic_index += 1


class PixelDsl(PixelProcessor):
    def process(self, **kwargs):
        self.gen_depth()
        self.gen_depth_color()
        self.gen_normal()
        self.gen_albedo()
        self.gen_instance()
        self.gen_instance_color()
        self.gen_instance_mask()
        self.gen_semantic()


class Structure(StructureProcessor):
    def process(self, *args, **kwargs):
        pass


class Render(RenderProcessor):
    def process(self, *args, **kwargs):
        self.gen_rgb()
