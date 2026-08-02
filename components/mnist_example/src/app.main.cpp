#include <SDL_main.h>
#include <mnist/app.hpp>

int main(int argc, char **argv)
{
	mnist::MNISTApp app{ argc, argv };
	return app.run();
}
