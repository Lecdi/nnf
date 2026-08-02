#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <backends/imgui_impl_sdl2.h>
#include <nnf/ml/models.hpp>
#include <nnf/utils/vector.hpp>

namespace mnist
{
    class MNISTApp
    {
    public:
    	MNISTApp(int argc, char **argv);
    
    	int run();
    	void exit(int exit_code = 0);
    
    private:
    	bool init_sdl();
    	bool init_imgui();
    	bool init_model();
    	void mainloop();
    	void cleanup_imgui();
    	void cleanup_sdl();
    
    	void process_events();
    	void respond_to_input();
    	int predict_digit(nnf::VectorInput<float> data);
    	void clear_canvas();
    	void update_canvas_after_pixel_click(int pixel_x, int pixel_y);
    	void render_canvas_sdl();
    	void render_frame_imgui();
    	void present_render();
    
    	static void brush_strong(uint8_t &pixel_value);
    	static void brush_medium(uint8_t &pixel_value);
    	static void brush_weak(uint8_t &pixel_value);
    
    	int exit_code_ = 0;
    	bool running_ = false;
    
    	int cell_size_ = 20;
    	int canvas_start_pixel_x_ = 0;
    	int canvas_start_pixel_y_ = 0;
    	std::array<uint8_t, 28 * 28> canvas_{};
    
    	bool mouse_down_ = false;
    	int predicted_ = -1;
    
    	std::unique_ptr<nnf::ml::Model> model_;
    	SDL_Window *window_;
    	SDL_Renderer *renderer_;
    };
}
