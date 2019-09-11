atlas_subdir(ParametricNet)

find_package(lwtnn)

atlas_add_library(ParametricNet
    ParametricNet/*.h Root/*.h Root/*.cxx
    PUBLIC_HEADERS ParametricNet
    PRIVATE_INCLUDE_DIRS ${LWTNN_INCLUDE_DIRS}
    PRIVATE_LINK_LIBRARIES ${LWTNN_LIBRARIES}
)

atlas_install_generic(data/*
    DESTINATION data PKGNAME_SUBDIR)
