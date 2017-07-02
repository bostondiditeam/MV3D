#include <stdio.h>
#include <iostream>
#include <new>

#include "classA.h"

extern "C"
{   
    void * ClassA( void )
    {
        // Note: Inside the function body, I can use C++. 
        return new(std::nothrow) classA;
    }

    void initializeStateVector( void * ptr, const void * meas_package_input)
    {
        classA * ref = reinterpret_cast<classA *>(ptr);
        ref->initializeStateVector(meas_package_input);
    }

    void predict(void * ptr, const void * predicted_state, const void * timeDiff)
    {
        classA * ref = reinterpret_cast<classA *>(ptr);
        ref->predict(predicted_state, timeDiff);
    }


    void update(void * ptr, const void * meas_package_input, const void * updated_state)
    {
        classA * ref = reinterpret_cast<classA *>(ptr);
        ref->update(meas_package_input, updated_state);
    }
}
