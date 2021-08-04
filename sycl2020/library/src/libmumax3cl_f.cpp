#include "libmumax3cl.hpp"

class Mumax3clUtil {
    public :
        Mumax3clUtil(int id) {
            this->obj = new Mumax3clUtil_t<float>(id);
        }
    private :
        Mumax3clUtil_t<float>* obj;
};
