module NetworkHandlers

function forward_pass_master(net, input)
    for layer in net
        input = layer(input)
    end
    return input
end

end