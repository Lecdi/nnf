#include <mnist/app.hpp>

#include <algorithm>
#include <cstdint>
#include <ios>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <utility>
#include <SDL.h>
#include <SDL_events.h>
#include <SDL_filesystem.h>
#include <SDL_mouse.h>
#include <SDL_rect.h>
#include <SDL_render.h>
#include <SDL_video.h>
#include <imgui.h>
#include <backends/imgui_impl_sdl2.h>
#include <backends/imgui_impl_sdlrenderer2.h>
#include <nnf/io/logging.hpp>
#include <nnf/ml/layers.hpp>
#include <nnf/ml/losses.hpp>
#include <nnf/ml/models.hpp>
#include <nnf/tensor/tensor.hpp>
#include <nnf/utils/vector.hpp>
#include <mnist/construct_model.hpp>

namespace mnist
{
    MNISTApp::MNISTApp(int argc, char **argv) : renderer_{ nullptr }, window_{ nullptr }
    {
    }
    
    int MNISTApp::run()
    {
        running_ = true;
    
        if (!init_sdl())
        {
            return -1;
        }
    
        if (!init_imgui())
        {
            cleanup_sdl();
            return -1;
        }
    
        if (!init_model())
        {
            cleanup_imgui();
            cleanup_sdl();
            return -1;
        }
    
        mainloop();
    
        cleanup_imgui();
        cleanup_sdl();
        return exit_code_;
    }
    
    void MNISTApp::exit(int exit_code)
    {
        exit_code_ = exit_code;
        running_ = false;
    }
    
    bool MNISTApp::init_sdl()
    {
        if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER | SDL_INIT_GAMECONTROLLER) != 0)
            return false;
    
        window_ = SDL_CreateWindow(
            "MNIST Model Tester",
            SDL_WINDOWPOS_CENTERED,
            SDL_WINDOWPOS_CENTERED,
            800,
            600,
            SDL_WINDOW_RESIZABLE
        );
    
        renderer_ = SDL_CreateRenderer(
            window_,
            -1,
            SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC
        );
    
        return true;
    }
    
    bool MNISTApp::init_imgui()
    {
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGui::StyleColorsDark();
    
        ImGui_ImplSDL2_InitForSDLRenderer(window_, renderer_);
        ImGui_ImplSDLRenderer2_Init(renderer_);
    
        return true;
    }
    
    bool MNISTApp::init_model()
    {
        try
        {
            model_ = construct_model();
            
            auto model_file = std::filesystem::path(SDL_GetBasePath()) / MODEL_RELATIVE_PATH;
            
            model_->compile();
            model_->load_from(model_file);
        }
        catch (const std::ios_base::failure &e)
        {
            return false;
        }
        catch (const std::runtime_error &e)
        {
            return false;
        }
    
        return true;
    }
    
    void MNISTApp::mainloop()
    {
        while (running_)
        {
            process_events();
            respond_to_input();
            render_canvas_sdl();
            render_frame_imgui();
            present_render();
        }
    }
    
    void MNISTApp::cleanup_imgui()
    {
        ImGui_ImplSDLRenderer2_Shutdown();
        ImGui_ImplSDL2_Shutdown();
        ImGui::DestroyContext();
    }
    
    void MNISTApp::cleanup_sdl()
    {
        SDL_DestroyRenderer(renderer_);
        SDL_DestroyWindow(window_);
        SDL_Quit();
    }
    
    void MNISTApp::process_events()
    {
        SDL_Event event;
    
        while (SDL_PollEvent(&event))
        {
            ImGui_ImplSDL2_ProcessEvent(&event);
    
            if (event.type == SDL_QUIT)
                exit(0);
    
            if (event.type == SDL_MOUSEBUTTONDOWN)
                if (event.button.button == SDL_BUTTON_LEFT)
                    mouse_down_ = true;
    
            if (event.type == SDL_MOUSEBUTTONUP)
                if (event.button.button == SDL_BUTTON_LEFT)
                    mouse_down_ = false;
        }
    }
    
    void MNISTApp::respond_to_input()
    {
        int mouse_pixel_x, mouse_pixel_y;
        SDL_GetMouseState(&mouse_pixel_x, &mouse_pixel_y);
    
        if (mouse_down_)
            update_canvas_after_pixel_click(mouse_pixel_x, mouse_pixel_y);
    }
    
    int MNISTApp::predict_digit(nnf::VectorInput<float> data)
    {
        return model_->predict(nnf::Tensor::from_data({ 28 * 28 }, std::move(data))).maxpos()[0];
    }
    
    void MNISTApp::clear_canvas()
    {
        std::fill(canvas_.begin(), canvas_.end(), 0);
    }
    
    void MNISTApp::update_canvas_after_pixel_click(int pixel_x, int pixel_y)
    {
        int canvas_x = (pixel_x - canvas_start_pixel_x_) / cell_size_;
        int canvas_y = (pixel_y - canvas_start_pixel_y_) / cell_size_;
    
        if (0 <= canvas_x && canvas_x < 28 && 0 <= canvas_y && canvas_y < 28)
        {
            brush_strong(canvas_[canvas_y + 28 * canvas_x]);
    
            if (canvas_x > 0)
                brush_medium(canvas_[canvas_y + 28 * canvas_x - 28]);
            if (canvas_x < 28 - 1)
                brush_medium(canvas_[canvas_y + 28 * canvas_x + 28]);
            if (canvas_y > 0)
                brush_medium(canvas_[canvas_y + 28 * canvas_x - 1]);
            if (canvas_y < 28 - 1)
                brush_medium(canvas_[canvas_y + 28 * canvas_x + 1]);
    
            if (canvas_x > 0 && canvas_y > 0)
                brush_weak(canvas_[canvas_y + 28 * canvas_x - 29]);
            if (canvas_x > 0 && canvas_y < 28 - 1)
                brush_weak(canvas_[canvas_y + 28 * canvas_x - 27]);
            if (canvas_x < 28 - 1 && canvas_y > 0)
                brush_weak(canvas_[canvas_y + 28 * canvas_x + 27]);
            if (canvas_x < 28 - 1 && canvas_y < 28 - 1)
                brush_weak(canvas_[canvas_y + 28 * canvas_x + 29]);
        }
    }
    
    void MNISTApp::render_canvas_sdl()
    {
        SDL_SetRenderDrawColor(renderer_, 0, 0, 0, 255);
        SDL_RenderClear(renderer_);
    
        for (int canvas_y = 0; canvas_y < 28; ++canvas_y)
            for (int canvas_x = 0; canvas_x < 28; ++canvas_x)
            {
                uint8_t value = canvas_[canvas_y + 28 * canvas_x];
    
                SDL_SetRenderDrawColor(renderer_, value, value, value, 255);
    
                SDL_Rect rect{};
                rect.x = canvas_x * cell_size_ + canvas_start_pixel_x_;
                rect.y = canvas_y * cell_size_ + canvas_start_pixel_y_;
                rect.w = cell_size_;
                rect.h = cell_size_;
    
                SDL_RenderFillRect(renderer_, &rect);
            }
    }
    
    void MNISTApp::render_frame_imgui()
    {
        ImGui_ImplSDLRenderer2_NewFrame();
        ImGui_ImplSDL2_NewFrame();
        ImGui::NewFrame();
    
        ImGui::Begin("Controls");
    
        if (ImGui::Button("Predict"))
        {
            auto data = nnf::Vector<float>(28 * 28);
            for (int i = 0; i < 28 * 28; ++i)
                data[i] = static_cast<float>(canvas_[i]) / 255.f;
            predicted_ = predict_digit(std::move(data));
        }
    
        if (ImGui::Button("Clear"))
        {
            clear_canvas();
            predicted_ = -1;
        }
    
        if (predicted_ >= 0)
            ImGui::Text("Prediction: %d", predicted_);
        else
            ImGui::Text("Draw a digit and press Predict");
    
        ImGui::End();
    
        ImGui::Render();
    }
    
    void MNISTApp::present_render()
    {
        ImGui_ImplSDLRenderer2_RenderDrawData(ImGui::GetDrawData(), renderer_);
        SDL_RenderPresent(renderer_);
    }
    
    void MNISTApp::brush_strong(uint8_t &pixel_value)
    {
        if (pixel_value < 240) pixel_value = 240;
    }
    
    void MNISTApp::brush_medium(uint8_t &pixel_value)
    {
        if (pixel_value < 150) pixel_value = 150;
    }
    
    void MNISTApp::brush_weak(uint8_t &pixel_value)
    {
        if (pixel_value < 90) pixel_value = 90;
    }
}
